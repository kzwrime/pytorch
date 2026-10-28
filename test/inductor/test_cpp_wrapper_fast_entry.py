# Owner(s): ["module: inductor"]
"""cpp_wrapper JIT entry through torch._C._aoti.InductorEntry.

config.cpp_wrapper_fast_entry replaces the generated Python wrapper's
per-input conversion with one C++ call; with
config.cpp_wrapper_borrow_static_inputs the graph borrows its static inputs
and InductorEntry keeps their handles across calls.
"""

import gc
import weakref

import torch
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def _compile(fn, example_inputs, static_input_idxs=()):
    gm = make_fx(fn)(*example_inputs)
    compiled = compile_fx_inner(
        gm,
        list(example_inputs),
        static_input_idxs=list(static_input_idxs),
        cpp_wrapper=True,
        is_inference=True,
    )
    return compiled, gm


def _call(compiled, args):
    out = compiled(list(args))
    return out if isinstance(out, (list, tuple)) else [out]


@instantiate_parametrized_tests
class TestCppWrapperFastEntry(TestCase):
    def setUp(self):
        super().setUp()
        self.patcher = config.patch(fx_graph_cache=False)
        self.patcher.__enter__()

    def tearDown(self):
        self.patcher.__exit__(None, None, None)
        super().tearDown()

    @parametrize("borrow", [False, True])
    def test_matches_eager(self, borrow):
        def fn(x, w, b):
            return torch.relu(x @ w + b), (x * 2).sum(1)

        x, w, b = torch.randn(4, 8), torch.randn(8, 8), torch.randn(8)
        with config.patch(cpp_wrapper_borrow_static_inputs=borrow):
            (compiled, gm), code = run_and_get_cpp_code(
                _compile, fn, (x, w, b), static_input_idxs=(1, 2)
            )
        self.assertIn("InductorEntry", code)
        self.assertEqual(
            "steal_or_borrow_from_raw_handles_to_raii_handles" in code, borrow
        )
        for _ in range(3):
            x = torch.randn(4, 8)
            expected = gm(x, w, b)
            args = [x, w, b]
            out = compiled(args)
            # Cleared before the graph runs, like the generic wrapper.
            self.assertEqual(args, [])
            for a, e in zip(out, expected):
                self.assertEqual(a, e)

    def test_disabled_keeps_generic_wrapper(self):
        def fn(x, w):
            return (x + w,)

        x, w = torch.randn(4), torch.randn(4)
        with config.patch(cpp_wrapper_fast_entry=False):
            (compiled, _), code = run_and_get_cpp_code(
                _compile, fn, (x, w), static_input_idxs=(1,)
            )
        self.assertNotIn("InductorEntry", code)
        self.assertNotIn("steal_or_borrow", code)
        self.assertEqual(_call(compiled, [x, w])[0], x + w)

    def test_static_input_is_renewed_and_released(self):
        def fn(x, w):
            return (x * w,)

        x = torch.randn(4)
        w1 = torch.randn(4)
        compiled, _ = _compile(fn, (x, w1), static_input_idxs=(1,))
        self.assertEqual(_call(compiled, [x, w1])[0], x * w1)
        self.assertEqual(_call(compiled, [x, w1])[0], x * w1)

        # Another tensor object at a static position (e.g. reloaded weights).
        w2 = torch.randn(4)
        self.assertEqual(_call(compiled, [x, w2])[0], x * w2)
        ref1 = weakref.ref(w1)
        del w1
        gc.collect()
        self.assertIsNone(ref1(), "the replaced static input is still held")
        # The renewed handle is reused.
        self.assertEqual(_call(compiled, [x, w2])[0], x * w2)

    def test_varying_static_input_gets_per_call_handles(self):
        def fn(x, w):
            return (x + w,)

        x = torch.randn(4)
        compiled, _ = _compile(fn, (x, torch.randn(4)), static_input_idxs=(1,))
        refs = []
        for _ in range(5):
            w = torch.randn(4)
            self.assertEqual(_call(compiled, [x, w])[0], x + w)
            refs.append(weakref.ref(w))
            del w
        gc.collect()
        # After two consecutive renewals the position is no longer cached.
        self.assertTrue(all(r() is None for r in refs[2:]))

    def test_in_place_update_of_static_input_is_seen(self):
        def fn(x, w):
            return (x + w,)

        x, w = torch.randn(4), torch.randn(4)
        compiled, _ = _compile(fn, (x, w), static_input_idxs=(1,))
        self.assertEqual(_call(compiled, [x, w])[0], x + w)
        w.copy_(torch.randn(4))
        self.assertEqual(_call(compiled, [x, w])[0], x + w)
        w.data = torch.randn(4)
        self.assertEqual(_call(compiled, [x, w])[0], x + w)

    def test_returned_static_input_is_a_new_handle(self):
        def fn(x, w):
            return x + 1, w

        x, w = torch.randn(4), torch.randn(4)
        compiled, _ = _compile(fn, (x, w), static_input_idxs=(1,))
        for _ in range(3):
            out = _call(compiled, [x, w])
            self.assertEqual(out[0], x + 1)
            self.assertEqual(out[1], w)
            self.assertEqual(out[1].data_ptr(), w.data_ptr())
            del out
            gc.collect()
        # The borrowed handle survived returning it three times.
        self.assertEqual(_call(compiled, [x, w])[1], w)

    def test_scalars_and_dynamic_shapes(self):
        def fn(x, w):
            return (x * w + x.shape[0],)

        x, w = torch.randn(5, 3), torch.randn(3)
        gm = make_fx(fn, tracing_mode="symbolic")(x, w)
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        example = [p.meta["val"] for p in placeholders]
        compiled = compile_fx_inner(
            gm, example, static_input_idxs=[1], cpp_wrapper=True, is_inference=True
        )
        for rows in (2, 5, 9):
            x = torch.randn(rows, 3)
            self.assertEqual(_call(compiled, [x, w])[0], x * w + rows)

    @config.patch(cpp_wrapper=True)
    def test_parameters_stay_static_with_synthetic_bases(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(4), requires_grad=False)

            def forward(self, a, b):
                a.add_(1)
                b.mul_(2)
                return a + b + self.w

        model = Model()
        compiled = torch.compile(model, backend="inductor", fullgraph=True)

        def run():
            base = torch.zeros(6)
            # Overlapping mutated aliases: AOTAutograd merges them into a
            # synthetic base input.
            out = compiled(base[:4], base[2:])
            return out, base

        with torch.no_grad():
            (out, base), code = run_and_get_cpp_code(run)
            self.assertIn("steal_or_borrow_from_raw_handles_to_raii_handles", code)
            ref_base = torch.zeros(6)
            ref = model(ref_base[:4], ref_base[2:])
            self.assertEqual(out, ref)
            self.assertEqual(base, ref_base)

    def test_inductor_entry_converts_like_the_generic_wrapper(self):
        def fn(a, b):
            return (a * b,)

        compiled, _ = _compile(fn, (torch.randn(3), torch.randn(3)))
        entry = torch._C._aoti.InductorEntry
        self.assertTrue(callable(entry))
        # A conversion error leaves args untouched and the entry uncalled.
        args = [torch.randn(3), object()]
        with self.assertRaises(RuntimeError):
            entry(1, 0, [])(args)
        self.assertEqual(len(args), 2)


if __name__ == "__main__":
    run_tests()
