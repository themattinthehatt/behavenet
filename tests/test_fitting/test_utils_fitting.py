import pytest


class TestClass:

    hparams = {
        'test1': 1,
        'test2': 2}

    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):

        assert 'test1' in self.hparams.keys()
        # assert 'test3' in self.hparams.keys()


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()
