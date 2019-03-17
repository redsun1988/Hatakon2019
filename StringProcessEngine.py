import re

class StringProcessEngine:
    def RemoveLoops(self, string):
        while re.search(r'\b(.+)(\s+\1\b)+', string):
            string = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', string)
        return string

    def clean(self, string):
        string = string.replace('<br>', ' ').replace('<br />', ' ').replace('&nbsp;', ' ')
        string = re.sub('<[^<]+?>', ' ', string)
        string = string.replace('\n', ' ').replace('\t', ' ').replace("\r", " ").strip()
        return string

#region Tests (kind of tests ^_^ )
if __name__ == "__main__":
   engine = StringProcessEngine()
   loopedStrings = ["test aaa aaa aaa"]
   targetStrings = ["test aaa"]

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
#endregion