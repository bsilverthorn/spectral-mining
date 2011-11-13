import plac
import specmine.tools.run_tests

if __name__ == "__main__":
    plac.call(specmine.tools.run_tests.main)

import nose

@plac.annotations()
def main():
    nose.main("specmine", argv = ["nose", "-v"])

