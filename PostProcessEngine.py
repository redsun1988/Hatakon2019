class PostProcessEngine:
    def RemoveLoops(self, string):
        return string

#Tests (kind of tests ^_^ )
engine = PostProcessEngine()
loopedStrings = ["test aaa aaa"]
targetStrings = ["test"]

lStringCount = len(loopedStrings)
tStringCount = len(targetStrings)

if (lStringCount != tStringCount):
    print("+++++++++++++++++++++++++++++++++++++WARNING++++++++++++++++++++++++++++++++++++++")
    print("the loopedStrings' count value does not equal with the targetStrings's count value")

for index in range(0, min(lStringCount, tStringCount)):
    result = engine.RemoveLoops(loopedStrings[index])
    if (result == targetStrings[index]):
        print("the %s value is OK!!!", (loopedStrings[index]))
    else:
        print("the '%s' value is NOT OK!!! The current result is '%s'. Should be '%s'" %
         (loopedStrings[index], result, targetStrings[index]))
