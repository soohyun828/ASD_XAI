# introduction
Human Figure Drawing(HFD) test를 할 때, 예를 들어 “머리가 크고 손을 작게 그려서 ASD이다.” 라는 식으로 평가를 하는데, 어느 정도가 큰 건지 객관적인 평가 기준이 명확히 정해져 있지 않아 이걸 딥러닝 모델로 판별해보고, 판별 결과를 설명할 방법을 찾아보고자 프로젝트를 진행하였다.<br> 
ASD(자폐스펙트럼) 아동과 TD(정상 아동)가 그린 HFD 데이터를 분류하는 task를 수행하였다.
![onealog](/assets/intro.png)   

### Data preprocessing
- bounding box로 HFD 이미지의 part별 annotation
- annotation 데이터로 전체 스케치에서 각 파트가 차지하는 면적 비율 구함 `relev_part_size_and_asd.ipynb`

# Framework
- HFD 이미지와 part 별 비율을 input으로 각각의 Encoder에서 embedding vector 뽑아 concat
- concat한 벡터를 input으로 하여 Transformer Encoder로 ASD/TD 판별
![onealog](/assets/model.png)

# Experiments
## Demo
- <strong>Training model</strong>
  ```sh
  python -u /data/psh68380/repos/ASD_capstone/main.py \
  --data_root "/local_datasets/ASD/asd_ver2_all_5folds_annotation" \
  --annotations_root "/data/psh68380/repos/ASD_capstone/part_proportion.csv" \
  --num_epochs 50 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --image_model "efficientnetb0" \
  --part_model "linear"
  ```
- <strong>Extract shapley value</strong>
  ```sh
  python -u /data/psh68380/repos/ASD_capstone/extract_shap_mine.py \
  --data_root "/local_datasets/ASD/asd_ver2_all_5folds_annotation" \
  --shap_save_root "/data/psh68380/repos/ASD_capstone/asd_ratio/class_shap.pkl" \
  --model "mine"
  ```

# Conclusion
![onealog](/assets/experiment1.png)
![onealog](/assets/experiment2.png)
