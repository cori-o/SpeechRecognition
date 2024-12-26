import re 

exclude_word = ["MBC", "뉴스", "좋아요", "구독"]
pattern = r'(^|\s|[^가-힣a-zA-Z0-9])(' + '|'.join(map(re.escape, exclude_word)) + r')($|\s|[^가-힣a-zA-Z0-9])'

text = "MBC 뉴스스 이덕영입니다."

if re.search(pattern, text):
    print('matched')
else:
    print('not matched')