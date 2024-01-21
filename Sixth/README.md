# 大模型评测教程

## 基础作业

使用 OpenCompass 评测 InternLM2-Chat-7B 模型在 C-Eval 数据集上的性能

分别评估了 InternLM2-Chat-7B 和 InternLM-Chat-7B，做了一下对比。

BTW. 该作业需要 40G 显存的实例。

### InternLM-Chat-7B

```
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_model_repos_internlm-chat-7b
----------------------------------------------  ---------  -------------  ------  -------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                         31.58
ceval-operating_system                          1c2571     accuracy       gen                                                                         36.84
ceval-computer_architecture                     a74dad     accuracy       gen                                                                         28.57
ceval-college_programming                       4ca32a     accuracy       gen                                                                         32.43
ceval-college_physics                           963fa8     accuracy       gen                                                                         26.32
ceval-college_chemistry                         e78857     accuracy       gen                                                                         16.67
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                         21.05
ceval-probability_and_statistics                65e812     accuracy       gen                                                                         38.89
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                         18.75
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                         35.14
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                         50
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                         22.22
ceval-high_school_physics                       adf25f     accuracy       gen                                                                         31.58
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                         15.79
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                         36.84
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                         26.32
ceval-middle_school_biology                     86817c     accuracy       gen                                                                         61.9
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                         63.16
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                         60
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                         47.83
ceval-college_economics                         f3f4e6     accuracy       gen                                                                         41.82
ceval-business_administration                   c1614e     accuracy       gen                                                                         33.33
ceval-marxism                                   cf874c     accuracy       gen                                                                         68.42
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                         70.83
ceval-education_science                         591fee     accuracy       gen                                                                         58.62
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                         70.45
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                         26.32
ceval-high_school_geography                     865461     accuracy       gen                                                                         47.37
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                         52.38
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                         58.33
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                         73.91
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                         63.16
ceval-logic                                     f5b022     accuracy       gen                                                                         31.82
ceval-law                                       a110a1     accuracy       gen                                                                         25
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                         30.43
ceval-art_studies                               2a1300     accuracy       gen                                                                         60.61
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                         62.07
ceval-legal_professional                        ce8787     accuracy       gen                                                                         39.13
ceval-high_school_chinese                       315705     accuracy       gen                                                                         63.16
ceval-high_school_history                       7eb30a     accuracy       gen                                                                         70
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                         59.09
ceval-civil_servant                             87d061     accuracy       gen                                                                         53.19
ceval-sports_science                            70f27b     accuracy       gen                                                                         52.63
ceval-plant_protection                          8941f9     accuracy       gen                                                                         59.09
ceval-basic_medicine                            c409d6     accuracy       gen                                                                         47.37
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                         40.91
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                         45.65
ceval-accountant                                002837     accuracy       gen                                                                         26.53
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                         22.58
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                         64.52
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                         34.69
ceval-physician                                 6e277d     accuracy       gen                                                                         40.82
ceval-stem                                      -          naive_average  gen                                                                         35.09
ceval-social-science                            -          naive_average  gen                                                                         52.79
ceval-humanities                                -          naive_average  gen                                                                         52.58
ceval-other                                     -          naive_average  gen                                                                         44.36
ceval-hard                                      -          naive_average  gen                                                                         23.91
ceval                                           -          naive_average  gen                                                                         44.16
```


### InternLM2-Chat-7B

```
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-7b
----------------------------------------------  ---------  -------------  ------  -------------------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                                     47.37
ceval-operating_system                          1c2571     accuracy       gen                                                                                     57.89
ceval-computer_architecture                     a74dad     accuracy       gen                                                                                     42.86
ceval-college_programming                       4ca32a     accuracy       gen                                                                                     51.35
ceval-college_physics                           963fa8     accuracy       gen                                                                                     36.84
ceval-college_chemistry                         e78857     accuracy       gen                                                                                     33.33
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                                     15.79
ceval-probability_and_statistics                65e812     accuracy       gen                                                                                     27.78
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                                     18.75
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                                     40.54
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                                     58.33
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                                     44.44
ceval-high_school_physics                       adf25f     accuracy       gen                                                                                     47.37
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                                     52.63
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                                     26.32
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                                     26.32
ceval-middle_school_biology                     86817c     accuracy       gen                                                                                     66.67
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                                     57.89
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                                     95
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                                     39.13
ceval-college_economics                         f3f4e6     accuracy       gen                                                                                     47.27
ceval-business_administration                   c1614e     accuracy       gen                                                                                     51.52
ceval-marxism                                   cf874c     accuracy       gen                                                                                     84.21
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                                     70.83
ceval-education_science                         591fee     accuracy       gen                                                                                     72.41
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                                     79.55
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                                     21.05
ceval-high_school_geography                     865461     accuracy       gen                                                                                     47.37
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                                     42.86
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                                     58.33
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                                     65.22
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                                     89.47
ceval-logic                                     f5b022     accuracy       gen                                                                                     54.55
ceval-law                                       a110a1     accuracy       gen                                                                                     41.67
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                                     56.52
ceval-art_studies                               2a1300     accuracy       gen                                                                                     69.7
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                                     86.21
ceval-legal_professional                        ce8787     accuracy       gen                                                                                     43.48
ceval-high_school_chinese                       315705     accuracy       gen                                                                                     68.42
ceval-high_school_history                       7eb30a     accuracy       gen                                                                                     75
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                                     68.18
ceval-civil_servant                             87d061     accuracy       gen                                                                                     55.32
ceval-sports_science                            70f27b     accuracy       gen                                                                                     73.68
ceval-plant_protection                          8941f9     accuracy       gen                                                                                     77.27
ceval-basic_medicine                            c409d6     accuracy       gen                                                                                     63.16
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                                     45.45
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                                     58.7
ceval-accountant                                002837     accuracy       gen                                                                                     44.9
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                                     38.71
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                                     45.16
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                                     51.02
ceval-physician                                 6e277d     accuracy       gen                                                                                     51.02
ceval-stem                                      -          naive_average  gen                                                                                     44.33
ceval-social-science                            -          naive_average  gen                                                                                     57.54
ceval-humanities                                -          naive_average  gen                                                                                     65.31
ceval-other                                     -          naive_average  gen                                                                                     54.94
ceval-hard                                      -          naive_average  gen                                                                                     34.62
ceval                                           -          naive_average  gen                                                                                     53.55
```

## 对比

| 评估数据集                                     | InternLM-Chat-7B | InternLM2-Chat-7B | 对比   |
| ---------------------------------------------- | ---------------- | ----------------- | ------ |
| ceval-operating_system                         | 36.84            | 57.89             | 21.05  |
| ceval-computer_architecture                    | 28.57            | 42.86             | 14.29  |
| ceval-college_programming                      | 32.43            | 51.35             | 18.92  |
| ceval-college_physics                          | 26.32            | 36.84             | 10.52  |
| ceval-college_chemistry                        | 16.67            | 33.33             | 16.66  |
| ceval-advanced_mathematics                     | 21.05            | 15.79             | -5.26  |
| ceval-probability_and_statistics               | 38.89            | 27.78             | -11.11 |
| ceval-discrete_mathematics                     | 18.75            | 18.75             | 0      |
| ceval-electrical_engineer                      | 35.14            | 40.54             | 5.4    |
| ceval-metrology_engineer                       | 50               | 58.33             | 8.33   |
| ceval-high_school_mathematics                  | 22.22            | 44.44             | 22.22  |
| ceval-high_school_physics                      | 31.58            | 47.37             | 15.79  |
| ceval-high_school_chemistry                    | 15.79            | 52.63             | 36.84  |
| ceval-high_school_biology                      | 36.84            | 26.32             | -10.52 |
| ceval-middle_school_mathematics                | 26.32            | 26.32             | 0      |
| ceval-middle_school_biology                    | 61.9             | 66.67             | 4.77   |
| ceval-middle_school_physics                    | 63.16            | 57.89             | -5.27  |
| ceval-middle_school_chemistry                  | 60               | 95                | 35     |
| ceval-veterinary_medicine                      | 47.83            | 39.13             | -8.7   |
| ceval-college_economics                        | 41.82            | 47.27             | 5.45   |
| ceval-business_administration                  | 33.33            | 51.52             | 18.19  |
| ceval-marxism                                  | 68.42            | 84.21             | 15.79  |
| ceval-mao_zedong_thought                       | 70.83            | 70.83             | 0      |
| ceval-education_science                        | 58.62            | 72.41             | 13.79  |
| ceval-teacher_qualification                    | 70.45            | 79.55             | 9.1    |
| ceval-high_school_politics                     | 26.32            | 21.05             | -5.27  |
| ceval-high_school_geography                    | 47.37            | 47.37             | 0      |
| ceval-middle_school_politics                   | 52.38            | 42.86             | -9.52  |
| ceval-middle_school_geography                  | 58.33            | 58.33             | 0      |
| ceval-modern_chinese_history                   | 73.91            | 65.22             | -8.69  |
| ceval-ideological_and_moral_cultivation        | 63.16            | 89.47             | 26.31  |
| ceval-logic                                    | 31.82            | 54.55             | 22.73  |
| ceval-law                                      | 25               | 41.67             | 16.67  |
| ceval-chinese_language_and_literature          | 30.43            | 56.52             | 26.09  |
| ceval-art_studies                              | 60.61            | 69.7              | 9.09   |
| ceval-professional_tour_guide                  | 62.07            | 86.21             | 24.14  |
| ceval-legal_professional                       | 39.13            | 43.48             | 4.35   |
| ceval-high_school_chinese                      | 63.16            | 68.42             | 5.26   |
| ceval-high_school_history                      | 70               | 75                | 5      |
| ceval-middle_school_history                    | 59.09            | 68.18             | 9.09   |
| ceval-civil_servant                            | 53.19            | 55.32             | 2.13   |
| ceval-sports_science                           | 52.63            | 73.68             | 21.05  |
| ceval-plant_protection                         | 59.09            | 77.27             | 18.18  |
| ceval-basic_medicine                           | 47.37            | 63.16             | 15.79  |
| ceval-clinical_medicine                        | 40.91            | 45.45             | 4.54   |
| ceval-urban_and_rural_planner                  | 45.65            | 58.7              | 13.05  |
| ceval-accountant                               | 26.53            | 44.9              | 18.37  |
| ceval-fire_engineer                            | 22.58            | 38.71             | 16.13  |
| ceval-environmental_impact_assessment_engineer | 64.52            | 45.16             | -19.36 |
| ceval-tax_accountant                           | 34.69            | 51.02             | 16.33  |
| ceval-physician                                | 40.82            | 51.02             | 10.2   |
| ceval-stem                                     | 35.09            | 44.33             | 9.24   |
| ceval-social-science                           | 52.79            | 57.54             | 4.75   |
| ceval-humanities                               | 52.58            | 65.31             | 12.73  |
| ceval-other                                    | 44.36            | 54.94             | 10.58  |
| ceval-hard                                     | 23.91            | 34.62             | 10.71  |
| ceval                                          | 44.16            | 53.55             | 9.39   |

看起来好像大部分数据集效果 InternLM2 比较好，有一部分效果变差了。
