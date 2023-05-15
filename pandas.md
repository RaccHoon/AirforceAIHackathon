# Pandas & Dataframe

## pandas

Python에서는 데이터 분석을 위해 여러 라이브러리를 포함해야 하는데, Pandas는 이런 데이터 분석을 위한 패키지이다. Pandas에서 R의 dataframe 데이터 타입을 참고하여 만든 것이 pandas dataframe이다.  

Pandas dataframe은 테이블 형식의 데이터를 다룰때 사용하며, 칼럼, 데이터, 인덱스로 구성된다.  
이때 dataframe의 각 칼럼은 series이다.  

## dataframe

### dataframe 만들기
```python
import pandas as pd
import numpy as np

npArray=np.array([[1, 2, 3], [4, 5, 6]])

dataframeSample=pd.dataframe(npArray)
display(dataframeSample)
```

> 위와 같은 방식으로 dataframe을 만들 수 있으며, ndarray 타입 뿐만 아니라 리스트, 딕셔너리, 튜플 등의 형태도 dataframe으로 변형할 수 있다.

### dataframe.shape

```python
print(dataframe.shape)
```
> (row, column)형식으로 리턴된다.

### 행, 열 원소 찾기
```python
display(df['A'])  // 열 출력
display(df.iloc[0])  // 행 출력
display(df.loc[0])  // 행 출력
display(df.ix[0])  // 행 출력
display(df.ix[0]['A'])  // 원소 출력
```