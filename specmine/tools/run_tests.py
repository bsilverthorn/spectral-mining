import plac
import specmine

if __name__ == "__main__":
    plac.call(specmine.tools.run_tests)

import nose

@plac.annotations()
def main():
    nose.main(specmine)

