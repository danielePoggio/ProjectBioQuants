
def sequenceAlignment(s1, s2):
    minLenString = min(len(s1), len(s2))
    numberMatching = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            flagSubString = False
            if s1[i] == s2[j]:  # abbiamo trovato un possibile inizio di sottostringa
                flagSubString = True
                lenSS = 1
                while flagSubString and (i+lenSS < len(s1)) and (j+lenSS < len(s2)):
                    if s1[i + lenSS] != s2[j + lenSS]:
                        flagSubString = False
                        numberMatching = numberMatching + lenSS
                    else:
                        lenSS = lenSS + 1

    return numberMatching / minLenString
