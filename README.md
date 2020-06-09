## aipack
    본 패키지는 AI Package로써 딥러닝관련한 여러가지 유틸기능들을 포함합니다.

![](https://img.shields.io/badge/python-3.6.1-blue)
![](https://img.shields.io/badge/tensorflow-2.0.0-orange)


### 설치 방법

```sh
pip3 install aipack
```

### 사용 예제

- TensorflowPack.converter

```python
    from aipack import TensorflowPack as tfPack
    # frozen graph으로부터 concrete function 생성
    tfPack.converter.frozengraph2function('frozen_graph.pb',['x:0'],['Identity:0'],True)
```

- TensorflowPack.data
```python

    from aipack import TensorflowPack as tfPack
    from utilpack.util import PyImageUtil

    # 데이터셋  정의 
    dataset =[
        {
            'index': 0,
            'image':'/Users/hian/picture/hian.jpg',
            'score':3.4,
            'label':'hian',
        },
        {
            'index': 1,
            'image': '/Users/hian/picture/faces.jpg',
            'score': 3.4,
            'label': 'faces',
        }
    ]

    # 이미지 전처리함수 작성 (필수아님 )
    import cv2
    def preFunc(img_cv):
        img_cv = cv2.resize(img_cv,(300,300))
        img_cv = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)
        return img_cv
    
    # 데이터셋 저장 ( 이미지의경우 키값이 'image' 으로 넘기면 자동으로 load )
    tfPack.data.processing(dataset,dataset_name='dataset',output_directory='.', preprocessFunc=preFunc)
    
    # 데이터셋 로드
    tfrecord_path='dataset.tfrecord'
    raw_dataset, decode = tfPack.data.read_tfrecord(tfrecord_path)

    for raw_data in raw_dataset:
    
        # 학습 (그래프) 이 아닌경우 eager를 True로 줌으로써 eager execution 가능
        data = decode(raw_data,eager=True)
        
        # 데이터 확인
        print(data)
        [print(v) for k,v in data.items() if k!='image']
    
        # matplotlib.pyplot.plot(data['image']/255.)과 같음
        PyImageUtil.plot(data['image'],color_mode='rgb')
```

