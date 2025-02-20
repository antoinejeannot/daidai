from mypackage.main import greetings


def test_main():
    assert greetings() == "Hello, World!"
