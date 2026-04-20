from dataclasses import dataclass


@dataclass(frozen=True)
class Primitive:
    name: str
    nargs: int

    def __call__(self, *args, **options):
        assert len(args) == self.nargs
        return bind(self, *args, **options)


neg = Primitive("neg", 1)
add = Primitive("add", 2)
mul = Primitive("mul", 2)
relu = Primitive("relu", 1)


interpreter_stack = []


def bind(primitive, *args, **options):
    top_interpreter = interpreter_stack[-1]
    return top_interpreter.process_primitive(primitive, *args, **options)


class Interpreter:
    def process_primitive(self, primitive, *args, **options):
        pass


class Value:
    def __mul__(self, other):
        return mul(self, other)


class EvalInterpreter(Interpreter):
    def process_primitive(self, primitive, *args, **options):
        print(args)
        args = [a.value for a in args]
        rule = eval_rules[primitive]
        res = rule(*args, **options)
        return Array(res)


@dataclass
class Array(Value):
    value: float


eval_rules = {
    neg: lambda x: -x,
    add: lambda x, y: x + y,
    mul: lambda x, y: x * y,
    relu: lambda x: max(0, x),
}


if __name__ == "__main__":
    def nn(x):
        return relu(Array(2) * x) 

    interpreter_stack += [EvalInterpreter()]

    y = nn(Array(-5.0))
    print(f"{y=}")
