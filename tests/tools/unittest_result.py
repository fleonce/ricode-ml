# Source - https://stackoverflow.com/a/57073164
# Posted by tomsgd
# Retrieved 2026-03-19, License - CC BY-SA 4.0
# slightly adjusted with different names
import unittest


class SubtestCountingTestResult(unittest.TextTestResult):

    def addSubTest(self, test, subtest, outcome):
        # handle failures calling base class
        super(SubtestCountingTestResult, self).addSubTest(test, subtest, outcome)
        # add to total number of tests run
        if hasattr(subtest, "_subDescription"):
            if hasattr(self, "___subtest"):
                self.testsRun += 1
            else:
                self.stream.write("\n")
                setattr(self, "___subtest", None)

            if self.showAll:
                self.stream.write("\t\t")
                self.stream.write(" subtest with params ")
                self.stream.write(subtest._subDescription())
                self.stream.write(" ...\n")
                self.stream.flush()

    def addSuccess(self, test):
        print(test)
