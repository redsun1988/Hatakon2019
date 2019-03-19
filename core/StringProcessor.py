import re

class StringProcessor:
    #Removes looks seqensios from the input string
    @staticmethod
    def RemoveLoops(string):
        while re.search(r'\b(.+)(\s+\1\b)+', string):
            string = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', string)
        return string

    #Clean HTML tags from the input string
    @staticmethod
    def Clean(string):
        string = string.replace('<br>', ' ').replace('<br />', ' ').replace('&nbsp;', ' ')
        string = re.sub('<[^<]+?>', ' ', string)
        string = string.replace('\n', ' ').replace('\t', ' ').replace("\r", " ").strip()
        string = re.sub(' +', ' ', string)
        return string

    #Create the sorted list of unique characters using the input list of strings
    @staticmethod
    def GetUniqueChars(p_list):
       return sorted(list(set((''.join([''.join(set(p)) for p in p_list])))))

    @staticmethod
    def GetTokenIndex(vocabulary):
       return dict([(char, i) for i, char in enumerate(vocabulary)]) 

    @staticmethod
    def GetReversTokenIndex(vocabulary):
       return dict([(i, char) for char, i in enumerate(vocabulary)])

#region Tests (kind of tests ^_^ )
if __name__ == "__main__":
   #region RemoveLoops Test
   loopedStrings = ["test aaa aaa aaa"]
   targetStrings = ["test aaa"]

   lStringCount = len(loopedStrings)
   tStringCount = len(targetStrings)

   if (lStringCount != tStringCount):
       print("+++++++++++++++++++++++++++++++++++++WARNING++++++++++++++++++++++++++++++++++++++")
       print("the loopedStrings' count value does not equal with the targetStrings's count value")

   for index in range(0, min(lStringCount, tStringCount)):
       result = StringProcessor.RemoveLoops(loopedStrings[index])
       if (result == targetStrings[index]):
           print("the %s value is OK!!!", (loopedStrings[index]))
       else:
           print("the '%s' value is NOT OK!!! The current result is '%s'. Should be '%s'" %
            (loopedStrings[index], result, targetStrings[index]))

    #endregion

#endregion