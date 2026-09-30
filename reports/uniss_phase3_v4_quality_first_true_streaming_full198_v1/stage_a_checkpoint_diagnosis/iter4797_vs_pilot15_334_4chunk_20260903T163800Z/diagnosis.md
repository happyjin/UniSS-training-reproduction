# Stage A checkpoint free-running diagnosis

- Checkpoint: `/opt/dlami/nvme/neuhao/UniSS/checkpoints/uniss_phase3_v4_quality_first_true_streaming_full198_v1/stage_a_formal/stage_a_formal8_20260903T054603Z/iter_0004797`
- Evaluations: 1336
- CTC blank collapse: **False**
- AR final-only/empty collapse: **False**
- AR teacher-forced token accuracy: **0.9548**
- Weighted CTC blank ratio: **0.6986**
- Weighted streaming WER/CER: **0.1778**
- Weighted causal-full WER/CER: **0.1182**

| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |
|---:|---|---|---:|---:|---|---|---:|
| 160 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6867 | 75 | Epidance also suggests that the plant was present decades before its first collection | wer | 0.0769 |
| 320 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6519 | 78 | Epidance also suggests that the plant was present decades before its first collection | wer | 0.0769 |
| 640 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6361 | 83 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000069954 | 0.6297 | 82 | Evidence also suggests that the plant was present decades before its first collection | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7662 | 52 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7292 | 60 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7354 | 57 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000116798 | 0.7354 | 57 | He that considers too much will not bring anything to performance | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6727 | 50 | Then he's an old man He's going to spend a month in Africa | wer | 0.0769 |
| 320 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6455 | 54 | Then he's an old man He's going to spend the month in Africa | wer | 0.1538 |
| 640 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6045 | 56 | Then he's an old man He's going to spend the month in Africa | wer | 0.1538 |
| 1280 | causal_full_asr | CommonVoice_EN_0000126435 | 0.6227 | 53 | When he's an old man he's going to spend a month in Africa | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000160093 | 0.8000 | 27 | The press is on my t-negence sir said Palt | wer | 0.4444 |
| 320 | causal_full_asr | CommonVoice_EN_0000160093 | 0.7951 | 29 | The press is a mighty engine Sir said Palt | wer | 0.1111 |
| 640 | causal_full_asr | CommonVoice_EN_0000160093 | 0.7854 | 31 | The press is a mighty engine Sir said Palt | wer | 0.1111 |
| 1280 | causal_full_asr | CommonVoice_EN_0000160093 | 0.7756 | 33 | The press is a mighty engine Sir said Pott | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000262462 | 0.6945 | 54 | It is across the Columbia River from Wensham in Washington | wer | 0.2000 |
| 320 | causal_full_asr | CommonVoice_EN_0000262462 | 0.7055 | 52 | It is across the Columbia River from Wensham in Washington | wer | 0.2000 |
| 640 | causal_full_asr | CommonVoice_EN_0000262462 | 0.7127 | 54 | It is across the Columbia River from Watsam in Washington | wer | 0.2000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000262462 | 0.7091 | 57 | It is across the Columbia River from Watsam in Washington | wer | 0.2000 |
| 160 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6906 | 41 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6519 | 46 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6243 | 48 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000312479 | 0.6133 | 52 | The soul of the world is nourished by people's happiness | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000352714 | 0.7262 | 62 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 320 | causal_full_asr | CommonVoice_EN_0000352714 | 0.6935 | 69 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 640 | causal_full_asr | CommonVoice_EN_0000352714 | 0.7024 | 69 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 1280 | causal_full_asr | CommonVoice_EN_0000352714 | 0.7083 | 68 | Teams from Topeka, Kansas and Wichita, Kansas joined from the Western Association | wer | 0.1667 |
| 160 | causal_full_asr | CommonVoice_EN_0000430515 | 0.7579 | 50 | Later years he occasionally proved a determined lower order batsman | wer | 0.0909 |
| 320 | causal_full_asr | CommonVoice_EN_0000430515 | 0.7228 | 52 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000430515 | 0.7123 | 55 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000430515 | 0.6912 | 58 | In later years he occasionally proved a determined lower order batsman | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8267 | 26 | Wood is best from making chosen blacks | wer | 0.5000 |
| 320 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8000 | 28 | Wood is best for making chosen blacks | wer | 0.3750 |
| 640 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8133 | 29 | Wood is best for making chosen blacks | wer | 0.3750 |
| 1280 | causal_full_asr | CommonVoice_EN_0000471648 | 0.8000 | 31 | Wood is best for making chosen blacks | wer | 0.3750 |
| 160 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8346 | 15 | Nothing personal on it | wer | 0.2500 |
| 320 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8425 | 15 | Nothing personal in it | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8346 | 15 | Nothing personal in it | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000502585 | 0.8189 | 17 | Nothing personal in it | wer | 0.0000 |
| 160 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8643 | 28 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 320 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8107 | 40 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 640 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8214 | 38 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 1280 | causal_full_asr | CommonVoice_EN_0000593818 | 0.8107 | 38 | If it wasn't difficult it wouldn't be a problem | wer | 0.0000 |
| 160 | causal_full_asr | DailyTalk_0000009768 | 0.6667 | 43 | Bring us a bottle of rumi more teeny and red wine | wer | 0.3000 |
| 320 | causal_full_asr | DailyTalk_0000009768 | 0.6398 | 45 | Bring us the bottle of rumney more teeny and red wine | wer | 0.4000 |
| 640 | causal_full_asr | DailyTalk_0000009768 | 0.6505 | 44 | Bring us the bottle of rumi more teeny and red wine | wer | 0.4000 |
| 1280 | causal_full_asr | DailyTalk_0000009768 | 0.6720 | 43 | Bring us the bottle of Remy Martinez and Redwine | wer | 0.4000 |
| 160 | causal_full_asr | DailyTalk_0000010084 | 0.7188 | 27 | Good morning sir What can I do for you | wer | 0.1111 |
| 320 | causal_full_asr | DailyTalk_0000010084 | 0.7109 | 28 | Good morning sir What can I do for you | wer | 0.1111 |
| 640 | causal_full_asr | DailyTalk_0000010084 | 0.7109 | 28 | Good morning sir What can I do for you | wer | 0.1111 |
| 1280 | causal_full_asr | DailyTalk_0000010084 | 0.6953 | 28 | The morning sir what can I do for you | wer | 0.2222 |
| 160 | causal_full_asr | EN_B00013_S06748_W000028 | 0.7294 | 42 | I just skateboards said his dad it's two slippery so that I'm | wer | 0.6364 |
| 320 | causal_full_asr | EN_B00013_S06748_W000028 | 0.6835 | 45 | Ride your skateboard said his dad it's two slippers and Adam | wer | 0.2727 |
| 640 | causal_full_asr | EN_B00013_S06748_W000028 | 0.6789 | 46 | I just skateboard said his dad it's two slippers and Adam | wer | 0.4545 |
| 1280 | causal_full_asr | EN_B00013_S06748_W000028 | 0.7018 | 45 | Ride your skateboard said his dad it's two slippers and Adam | wer | 0.2727 |
| 160 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6948 | 49 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6808 | 47 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6620 | 47 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00036_S05339_W000016 | 0.6620 | 49 | To use the present perfect correctly you need to know things like | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00043_S01557_W000006 | 0.6686 | 40 | Guys they were rapidly using up this resource | wer | 0.1250 |
| 320 | causal_full_asr | EN_B00043_S01557_W000006 | 0.7035 | 39 | Guys they were rapidly using up this resource | wer | 0.1250 |
| 640 | causal_full_asr | EN_B00043_S01557_W000006 | 0.6453 | 39 | Because they were rapidly using up this resource | wer | 0.1250 |
| 1280 | causal_full_asr | EN_B00043_S01557_W000006 | 0.6512 | 39 | Because they were rapidly using up this resource | wer | 0.1250 |
| 160 | causal_full_asr | EN_B00043_S01623_W000011 | 0.7160 | 80 | I found all that I had to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0909 |
| 320 | causal_full_asr | EN_B00043_S01623_W000011 | 0.6942 | 89 | I found all that hard to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00043_S01623_W000011 | 0.7039 | 89 | I found all that hard to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00043_S01623_W000011 | 0.6869 | 90 | I found all that hard to believe too It must have been terrible And there was nothing anyone could do about it | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6606 | 90 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 320 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6483 | 88 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 640 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6177 | 92 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 1280 | causal_full_asr | EN_B00043_S01957_W000002 | 0.6330 | 90 | Now we're not talking about video games or computer generated actors which looks tremendously realistic | wer | 0.1429 |
| 160 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7288 | 92 | That's the image Let's go through the main door Turn F to walk down to the end of the car to the end of the car and it's the last car on the right | wer | 0.5385 |
| 320 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7076 | 97 | That's good Miss Just go through the main door Turn F work down to the end of the car to the end and it's the last car on the right | wer | 0.3462 |
| 640 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7076 | 99 | Certainly miss Just go through the main door Turn FET walk down to the end of the car to the end and it's the last car on the right | wer | 0.2308 |
| 1280 | causal_full_asr | EN_B00043_S02699_W000000 | 0.7119 | 100 | Certainly miss Just go through the main door Turn left walk down to the end of the car to the end and it's the last car on the right | wer | 0.1923 |
| 160 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7788 | 34 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7389 | 41 | Winder you store the cookies from the cookie jar | wer | 0.2222 |
| 640 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7478 | 41 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S01182_W000001 | 0.7345 | 38 | Linda you stole the cookies from the cookie jar | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6800 | 83 | So for those different categories of use you'll find that that modal verbs are used slightly differently | wer | 0.0588 |
| 320 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6343 | 82 | So for those different categories of use you'll find that the modal verbs are used slightly differently | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6229 | 83 | So for those different categories of use you'll find that the modal verbs are used slightly differently | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S03599_W000216 | 0.6371 | 79 | So for those different categories of use you'll find that the modal verbs are used slightly differently | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6471 | 44 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6235 | 46 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6235 | 47 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07041_W000493 | 0.6118 | 46 | Maybe they just didn't like me so they didn't want to talk to me | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7222 | 36 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S07870_W000001 | 0.6852 | 34 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7037 | 36 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S07870_W000001 | 0.7160 | 33 | I've never heard of him I doubt he's very powerful | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00048_S08821_W000047 | 0.7161 | 45 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00048_S08821_W000047 | 0.7203 | 44 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S08821_W000047 | 0.7203 | 44 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S08821_W000047 | 0.6907 | 46 | So after you noted the hours starting at zero which is midnight | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6609 | 144 | Every two months my coworkers and I would come together to discuss the new semester schedule Uhm meetings were usually held in the staff room at our institute | wer | 0.0357 |
| 320 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6401 | 141 | Every two months my coworkers and I would come together to discuss the new semester schedule Our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6263 | 146 | Every two months my coworkers and I would come together to discuss the new semester schedule Our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00048_S09601_W000039 | 0.6280 | 145 | Every two months my coworkers and I would come together to discuss the new semester schedule Our meetings were usually held in the staff room at our institute | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6327 | 134 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6135 | 135 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6135 | 139 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00052_S08813_W000015 | 0.6038 | 142 | We come from different parties but we're Americans first and our obligations to you must compel all of us Democrats and Republicans to cooperate and compromise | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00058_S01107_W000030 | 0.5943 | 57 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S01107_W000030 | 0.5714 | 55 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00058_S01107_W000030 | 0.5943 | 52 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00058_S01107_W000030 | 0.6171 | 51 | The entire project was dropped in my lap after Jason resigned | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5816 | 117 | So you original meeting that we did a bunch about the shows about of seeing people in the inner earth You said that took place in September | wer | 0.1538 |
| 320 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5765 | 129 | So your original meeting that we did a bunch about the shows about of seeing people in the inner earth You said that took place in September | wer | 0.1154 |
| 640 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5383 | 132 | So your original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that's a place in September | wer | 0.0769 |
| 1280 | causal_full_asr | EN_B00058_S06163_W000045 | 0.5357 | 129 | So your original meeting that we did a bunch of episodes about of seeing people in the inner earth You said that took place in September | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00058_S06426_W000037 | 0.7218 | 53 | It looks like this curve here where f here is on the horizontal axis | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00058_S06426_W000037 | 0.6972 | 52 | And let's say this curve here where f here is on the horizontal axis | wer | 0.2143 |
| 640 | causal_full_asr | EN_B00058_S06426_W000037 | 0.6761 | 55 | And looks like this curve here where f here is on the horizontal axis | wer | 0.0714 |
| 1280 | causal_full_asr | EN_B00058_S06426_W000037 | 0.6831 | 54 | It looks like this curve here where F here is on the horizontal axis | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5587 | 124 | In Tibetan Buddhism Mahlers are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 320 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5561 | 122 | In Tibetan Buddhism Mahlers are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 640 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5614 | 118 | In Tibetan Buddhism Mahlers are mainly used to count mantras These mantras can be recited for different purposes linked to working with mind | wer | 0.0435 |
| 1280 | causal_full_asr | EN_B00064_S09941_W000015 | 0.5666 | 119 | In Tibetan Buddhism Mahlers are mainly used to count montras These montras can be recited for different purposes linked to working with mind | wer | 0.1304 |
| 160 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6588 | 140 | In addition the journal covers signaling networks synthetic biology systems biology draw it discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 320 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6390 | 145 | In addition the journal covers signaling networks synthetic biology systems biology draw it discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 640 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6516 | 136 | In addition the journal covers signaling networks synthetic biology systems biology draw data discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 1280 | causal_full_asr | EN_B00083_S00712_W000002 | 0.6606 | 134 | In addition the journal covers signaling networks synthetic biology systems biology draw data discovery and computation and modeling of regulatory pathways | wer | 0.1000 |
| 160 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6363 | 243 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that would build success I've tried to do that throughout my relationship with them | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6107 | 257 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success I've tried to do that throughout my relationship with them | wer | 0.0175 |
| 640 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6096 | 259 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success I've tried to do that throughout my relationship with them | wer | 0.0175 |
| 1280 | causal_full_asr | EN_B00083_S03017_W000001 | 0.6055 | 259 | We don't learn from our successes and we don't learn from our failures in a way that allows the enterprise to grow and develop So I think it's essential that we treat people with that respect and trust and build systems around them that that would build success I've tried to do that throughout my relationship with them | wer | 0.0175 |
| 160 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6897 | 66 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 320 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6586 | 70 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 640 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6828 | 65 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 1280 | causal_full_asr | EN_B00083_S08368_W000007 | 0.6862 | 63 | And these are not the variables of algebra A variable is any characteristic | wer | 0.0000 |
| 160 | causal_full_asr | EN_B00083_S08530_W000016 | 0.7729 | 40 | You're unmute we're gonna cheer you Sorry Good morning everybody | wer | 0.5833 |
| 320 | causal_full_asr | EN_B00083_S08530_W000016 | 0.7131 | 49 | You're unmute We're gonna hear you Sorry Good morning everybody | wer | 0.5000 |
| 640 | causal_full_asr | EN_B00083_S08530_W000016 | 0.6932 | 50 | You're unmute We're not here You Sorry Good morning everybody | wer | 0.5833 |
| 1280 | causal_full_asr | EN_B00083_S08530_W000016 | 0.7012 | 48 | You're unmute We're not here You Sorry Good morning everybody | wer | 0.5833 |
| 160 | causal_full_asr | LibriSpeech_0000011154 | 0.6419 | 158 | Other villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.1471 |
| 320 | causal_full_asr | LibriSpeech_0000011154 | 0.6202 | 167 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 640 | causal_full_asr | LibriSpeech_0000011154 | 0.6295 | 159 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 1280 | causal_full_asr | LibriSpeech_0000011154 | 0.6202 | 161 | All the villagers said it was a fine idea So they stopped working for once and began to plan a celebration They thought that there ought to be swimming races and treefelling contests | wer | 0.0882 |
| 160 | causal_full_asr | LibriSpeech_0000011649 | 0.6062 | 172 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000011649 | 0.5900 | 172 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000011649 | 0.5914 | 173 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000011649 | 0.5796 | 176 | Or on several hills as Roman as well as Bostonian history testifies can only be guessed by its tribute in the form of the Blue Hills Reservation This state recreation park and forest reserve | wer | 0.0000 |
| 160 | causal_full_asr | LibriSpeech_0000192309 | 0.7035 | 62 | In the hope of gaining a little more time he repeated his question no | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000192309 | 0.6972 | 63 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 640 | causal_full_asr | LibriSpeech_0000192309 | 0.6814 | 63 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 1280 | causal_full_asr | LibriSpeech_0000192309 | 0.6845 | 64 | In the hope of gaining a little more time he repeated his question a no | wer | 0.0714 |
| 160 | causal_full_asr | LibriSpeech_0000214472 | 0.7765 | 111 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 320 | causal_full_asr | LibriSpeech_0000214472 | 0.7500 | 113 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 640 | causal_full_asr | LibriSpeech_0000214472 | 0.7206 | 117 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 1280 | causal_full_asr | LibriSpeech_0000214472 | 0.7191 | 119 | By common consent the subject is never mentioned between us The bitter irony of his tone thus far suddenly disappeared He spoke eagerly and anxiously | wer | 0.0000 |
| 160 | causal_full_asr | LibriSpeech_0000238204 | 0.7607 | 122 | What answer one of those ever tells Why men who advertise for wives can only be speedy adventurers The sort of person one reads of in books and that in needs and meal lines | wer | 0.3125 |
| 320 | causal_full_asr | LibriSpeech_0000238204 | 0.7292 | 132 | Would answer one of those advertisements Why men who advertise for whites can only be speedy adventurers the sort of person one reads of in books and knitting meets a meal on | wer | 0.1875 |
| 640 | causal_full_asr | LibriSpeech_0000238204 | 0.7235 | 132 | Would answer one of those advertisements Why men who advertise for wives can only be speedy adventurers the sort of person one reads of in books and knitting needs a meal | wer | 0.1875 |
| 1280 | causal_full_asr | LibriSpeech_0000238204 | 0.7149 | 134 | Would answer one of those evertastant Why men who advertise for wives can only be speedy adventurers the sort of person one reads of in books and they don't need some meal life | wer | 0.2188 |
| 160 | causal_full_asr | LibriSpeech_0000263750 | 0.7320 | 112 | A magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in walk Reggie | wer | 0.1000 |
| 320 | causal_full_asr | LibriSpeech_0000263750 | 0.7136 | 114 | In a magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in walk Reggie | wer | 0.1500 |
| 640 | causal_full_asr | LibriSpeech_0000263750 | 0.7119 | 115 | In a magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sedley he announced in walk Reggie | wer | 0.1500 |
| 1280 | causal_full_asr | LibriSpeech_0000263750 | 0.7018 | 116 | A magnificent person with powdered hair breeches and silk stockings presented himself Lord Reginald Sidley he announced in walk Reggie | wer | 0.0500 |
| 160 | causal_full_asr | LibriSpeech_0000271146 | 0.6321 | 101 | But it is full of angel leaving the mounds enormous bones which the Indians attribute some gigantic race which lived in a past age | wer | 0.3043 |
| 320 | causal_full_asr | LibriSpeech_0000271146 | 0.6244 | 106 | But it is full of Angela living the mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.2609 |
| 640 | causal_full_asr | LibriSpeech_0000271146 | 0.6347 | 101 | For it is full of anteliving mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.1304 |
| 1280 | causal_full_asr | LibriSpeech_0000271146 | 0.6244 | 105 | For it is full of anteliving the mounds enormous bones which the Indians attribute to some gigantic race which lived in a past age | wer | 0.1739 |
| 160 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.8528 | 19 | Just who's about the netting and that's | wer | 0.8571 |
| 320 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.8344 | 21 | Just two What's about the magnetic impact | wer | 0.5714 |
| 640 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.7914 | 22 | That's true What about the mentality in that | wer | 0.4286 |
| 1280 | causal_full_asr | NCSSD_R_EN_0000000402 | 0.7669 | 26 | That's true What about the genetic impact | wer | 0.1429 |
| 160 | causal_full_asr | VCTK_0000006143 | 0.7089 | 34 | Being captain of this club is fantastic | wer | 0.0000 |
| 320 | causal_full_asr | VCTK_0000006143 | 0.7025 | 34 | Being captain of this club is fantastic | wer | 0.0000 |
| 640 | causal_full_asr | VCTK_0000006143 | 0.6835 | 35 | Being captain of this club is fantastic | wer | 0.0000 |
| 1280 | causal_full_asr | VCTK_0000006143 | 0.6772 | 35 | Being captain of this club is fantastic | wer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0003940788 | 0.7815 | 92 | 呃然后呢这个因为它很大能帮助注意力同时性情也没比较凶猛就是特有那种王者之气 | cer | 0.1892 |
| 320 | causal_full_asr | emilia_zh_0003940788 | 0.7726 | 95 | 呃然后呢这个因为它很大呃能帮助助猎同时性情也没比较凶猛就是特有那种王者之气 | cer | 0.1351 |
| 640 | causal_full_asr | emilia_zh_0003940788 | 0.7726 | 93 | 呃然后呢这个因为它很大呃能帮助助练同时性情也没比较凶猛就是特有那种王者之气 | cer | 0.1351 |
| 1280 | causal_full_asr | emilia_zh_0003940788 | 0.7638 | 94 | 呃然后呢这个因为它很大呃能帮助助练同时性情也比较凶猛就是特有那种王者之气 | cer | 0.1081 |
| 160 | causal_full_asr | emilia_zh_0003942539 | 0.6812 | 145 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 320 | causal_full_asr | emilia_zh_0003942539 | 0.6614 | 152 | 其实我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0003942539 | 0.6554 | 156 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 1280 | causal_full_asr | emilia_zh_0003942539 | 0.6535 | 159 | 所以我觉得这个应该是需要一直练习下去的事情吧所以当他问我们的时候我也是觉得很惶恐的我不知道我可不可以把我的一些心得 | cer | 0.0526 |
| 160 | causal_full_asr | emilia_zh_0004036374 | 0.8022 | 95 | 则有关共产主义运动的具体策略和节目的叙述来充当马克思主义哲学发展的内在逻辑叙述 | cer | 0.1500 |
| 320 | causal_full_asr | emilia_zh_0004036374 | 0.7985 | 100 | 则有关共产主义运动的具体策略和结论的叙述来充当了马克思主义哲学发展的内在逻辑叙述 | cer | 0.0750 |
| 640 | causal_full_asr | emilia_zh_0004036374 | 0.7931 | 102 | 则有关共产主义运动的具体策略和结论的叙述来重担的马克思主义哲学发展的内在逻辑叙述 | cer | 0.1000 |
| 1280 | causal_full_asr | emilia_zh_0004036374 | 0.7913 | 103 | 则有关共产主义运动的具体策略和结论的叙述来重担的马克思主义哲学发展的内在逻辑叙述 | cer | 0.1000 |
| 160 | causal_full_asr | emilia_zh_0004213341 | 0.7176 | 67 | 世界上每天都有机器在发生这些奇迹大都来源于精神的力量 | cer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0004213341 | 0.7061 | 69 | 世界上每天都有奇迹在发生这些奇迹大多来源于精神的力量 | cer | 0.0385 |
| 640 | causal_full_asr | emilia_zh_0004213341 | 0.6985 | 73 | 世界上每天都有积极的发生这些奇迹大都来源于精神的力量 | cer | 0.1154 |
| 1280 | causal_full_asr | emilia_zh_0004213341 | 0.6870 | 76 | 世界上每天都有奇迹的发生这些奇迹大多来源于精神的力量 | cer | 0.0769 |
| 160 | causal_full_asr | emilia_zh_0004343392 | 0.8339 | 37 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004343392 | 0.8192 | 43 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004343392 | 0.8044 | 45 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004343392 | 0.8044 | 45 | 他还不走天黑了怎么办呢小红猴子说 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004422836 | 0.7768 | 128 | 精神训犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能坦白交代罪行杰西在斯打中失守之命按当时编区的法令 | cer | 0.1961 |
| 320 | causal_full_asr | emilia_zh_0004422836 | 0.7768 | 130 | 精神训犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能砍白胶带罪行且戏在斯打中失守致命按当时编区的法令 | cer | 0.2157 |
| 640 | causal_full_asr | emilia_zh_0004422836 | 0.7752 | 135 | 精神训犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能砍白胶带罪行且戏在斯打中失守致命按当时编区的法令 | cer | 0.2157 |
| 1280 | causal_full_asr | emilia_zh_0004422836 | 0.7691 | 136 | 精神讯犯罪嫌疑人对案情供认不会因四人军不满十八周岁尚能砍白胶带罪行且戏在四大中失守致命按当时编区的法令 | cer | 0.2157 |
| 160 | causal_full_asr | emilia_zh_0004472296 | 0.8292 | 49 | 里面的不对啊这是赵白书带我给你们念上念 | cer | 0.2105 |
| 320 | causal_full_asr | emilia_zh_0004472296 | 0.8385 | 47 | 明艳的不对啊这是赵白书带我给你的面纱您 | cer | 0.4211 |
| 640 | causal_full_asr | emilia_zh_0004472296 | 0.8323 | 50 | 明艳的不对啊这是赵白书带我跟你念上音念 | cer | 0.3684 |
| 1280 | causal_full_asr | emilia_zh_0004472296 | 0.8447 | 47 | 宁愿的不对啊这是赵白书待我给你念上音 | cer | 0.2632 |
| 160 | causal_full_asr | emilia_zh_0004621493 | 0.6905 | 44 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 320 | causal_full_asr | emilia_zh_0004621493 | 0.7190 | 41 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 640 | causal_full_asr | emilia_zh_0004621493 | 0.6381 | 49 | He is an attractive young man who steals a little bit here and there | wer | 0.2143 |
| 1280 | causal_full_asr | emilia_zh_0004621493 | 0.6286 | 49 | Here's an attractive young man who steals a little bit here and there | wer | 0.2857 |
| 160 | causal_full_asr | emilia_zh_0004633749 | 0.5743 | 82 | The pilot had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 320 | causal_full_asr | emilia_zh_0004633749 | 0.5060 | 87 | The parlor had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 640 | causal_full_asr | emilia_zh_0004633749 | 0.5382 | 84 | The pilot had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 1280 | causal_full_asr | emilia_zh_0004633749 | 0.5261 | 84 | The pilot had once been two rooms and the floor was swayed back where the partition had been cut away | wer | 0.3000 |
| 160 | causal_full_asr | emilia_zh_0004659501 | 0.8023 | 48 | And thought old you just write nine or ten And then where do where to | wer | 0.1429 |
| 320 | causal_full_asr | emilia_zh_0004659501 | 0.7794 | 48 | And thought old you just write nine or ten And then where do where to | wer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0004659501 | 0.8023 | 47 | And thought old you just write nine or ten And then where do where to | wer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0004659501 | 0.7937 | 48 | And for old you just write nine or ten And then where do where to | wer | 0.1429 |
| 160 | causal_full_asr | emilia_zh_0004665404 | 0.6543 | 82 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0004665404 | 0.6114 | 89 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0004665404 | 0.5971 | 90 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0004665404 | 0.5971 | 88 | If the wine be sweet I will drink it with him and if it be bitter I will drink it with him also was my answer | wer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0004665564 | 0.5639 | 155 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his acid of features He rolled with an iron hand in a velvet political glove | wer | 0.2286 |
| 320 | causal_full_asr | emilia_zh_0004665564 | 0.5442 | 163 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his hesitative features He ruled with an iron hand in a velvet political glove | wer | 0.2286 |
| 640 | causal_full_asr | emilia_zh_0004665564 | 0.5305 | 165 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his hesitative features He ruled with an iron hand in a velvet political glove | wer | 0.2286 |
| 1280 | causal_full_asr | emilia_zh_0004665564 | 0.5599 | 164 | But he ran an extremely efficient organization and he was not known ever to have fainted at the sight of blood despite his hesitating features He rolled with an iron hand in a velvet political glove | wer | 0.2000 |
| 160 | causal_full_asr | emilia_zh_0004692595 | 0.6711 | 72 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 320 | causal_full_asr | emilia_zh_0004692595 | 0.6312 | 77 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 640 | causal_full_asr | emilia_zh_0004692595 | 0.6146 | 80 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 1280 | causal_full_asr | emilia_zh_0004692595 | 0.6113 | 81 | Different situations and then later the differential equation interpreted in black diagram terms | wer | 0.0769 |
| 160 | causal_full_asr | emilia_zh_0004705832 | 0.6254 | 181 | These words are automatic we no longer visualize it which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live like kids | wer | 0.0833 |
| 320 | causal_full_asr | emilia_zh_0004705832 | 0.6193 | 183 | These words are automatic we no longer visualize it which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live like kids | wer | 0.0833 |
| 640 | causal_full_asr | emilia_zh_0004705832 | 0.6101 | 190 | These words are automatic we no longer visualize it which is why it takes kids longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live on kids | wer | 0.1111 |
| 1280 | causal_full_asr | emilia_zh_0004705832 | 0.6070 | 188 | These words are automatic we no longer visualize it which is why it takes kids are longer to think because they're still visualizing the words it's not automatic anymore which also explains why kids live on kids | wer | 0.0833 |
| 160 | causal_full_asr | emilia_zh_0004732654 | 0.5809 | 72 | But quite suddenly the fire ahead gave a pale flicker and went down and the clocking ceased | wer | 0.0588 |
| 320 | causal_full_asr | emilia_zh_0004732654 | 0.5643 | 74 | But quite suddenly the far ahead gave a pale flicker and went down and the clocking ceased | wer | 0.1176 |
| 640 | causal_full_asr | emilia_zh_0004732654 | 0.5560 | 74 | But quite suddenly the firehead gave a pale flicker and went down and the clocking ceased | wer | 0.1765 |
| 1280 | causal_full_asr | emilia_zh_0004732654 | 0.5436 | 74 | But quite suddenly the firehead gave a pale flicker and went down and the clocking ceased | wer | 0.1765 |
| 160 | causal_full_asr | emilia_zh_0004841554 | 0.6654 | 64 | I say excuse me but can you tell us whether purpose just come down from London | wer | 0.1176 |
| 320 | causal_full_asr | emilia_zh_0004841554 | 0.6342 | 66 | I say excuse me but can you tell us where the purpose just come down from London | wer | 0.1765 |
| 640 | causal_full_asr | emilia_zh_0004841554 | 0.6265 | 67 | I say excuse me but can you tell us where the purpose just come down from London | wer | 0.1765 |
| 1280 | causal_full_asr | emilia_zh_0004841554 | 0.6265 | 66 | I say excuse me but can you tell us where the purpose just come down from London | wer | 0.1765 |
| 160 | causal_full_asr | emilia_zh_0005058847 | 0.8443 | 74 | 因为磁性难起磁嘛就是磁背的磁纵然发起难得久停为什么呢人到自私 | cer | 0.2667 |
| 320 | causal_full_asr | emilia_zh_0005058847 | 0.8386 | 79 | 因为磁性难起磁嘛就是磁背的磁纵然发起难得久停为什么呢人到自私 | cer | 0.2667 |
| 640 | causal_full_asr | emilia_zh_0005058847 | 0.8330 | 81 | 因为磁性难起磁嘛就是磁背的磁纵然发起难得久停为什么呢人到自私 | cer | 0.2667 |
| 1280 | causal_full_asr | emilia_zh_0005058847 | 0.8368 | 80 | 因为磁性难起磁嘛就是磁碑的磁纵然发起难得久停为什么呢人道自私 | cer | 0.2333 |
| 160 | causal_full_asr | emilia_zh_0005070101 | 0.8058 | 109 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 320 | causal_full_asr | emilia_zh_0005070101 | 0.8043 | 118 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 640 | causal_full_asr | emilia_zh_0005070101 | 0.7997 | 115 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 1280 | causal_full_asr | emilia_zh_0005070101 | 0.8012 | 117 | 就可能是法国人比任何其他民族都更不适合在专制制度的原址上建立一个和平而自由的法治国家 | cer | 0.0238 |
| 160 | causal_full_asr | emilia_zh_0005070340 | 0.7825 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 320 | causal_full_asr | emilia_zh_0005070340 | 0.7855 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 640 | causal_full_asr | emilia_zh_0005070340 | 0.7734 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 1280 | causal_full_asr | emilia_zh_0005070340 | 0.7764 | 65 | 从事了一下名副其实的哲学工作因为他继续着前人的努力 | cer | 0.0400 |
| 160 | causal_full_asr | emilia_zh_0005184596 | 0.7071 | 54 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005184596 | 0.7071 | 54 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005184596 | 0.7020 | 54 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005184596 | 0.7020 | 55 | 那也不能对你两只手这样的姿势有什么批评 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0005245611 | 0.7458 | 69 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 320 | causal_full_asr | emilia_zh_0005245611 | 0.7322 | 71 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 640 | causal_full_asr | emilia_zh_0005245611 | 0.7288 | 71 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 1280 | causal_full_asr | emilia_zh_0005245611 | 0.7288 | 71 | 他提出预祝的商品只有再次革新才能更好的满足住人的需求 | cer | 0.0741 |
| 160 | causal_full_asr | emilia_zh_0005313767 | 0.7912 | 47 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005313767 | 0.7766 | 55 | 您有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0526 |
| 640 | causal_full_asr | emilia_zh_0005313767 | 0.7875 | 52 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005313767 | 0.7912 | 52 | 你有五秒钟的时间阅读第一小题的有关内容 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0005347375 | 0.7212 | 57 | 但我们却不知道他是什么只能确定他存在一定效应 | cer | 0.1818 |
| 320 | causal_full_asr | emilia_zh_0005347375 | 0.7389 | 56 | 但我们却不知道他是什么只能确定他存在一粒小叶 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0005347375 | 0.7301 | 59 | 但我们却不知道他是什么只能确定他存在以利效应 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0005347375 | 0.7301 | 60 | 但我们却不知道他是什么只能确定他存在引力效应 | cer | 0.0909 |
| 160 | causal_full_asr | emilia_zh_0005600573 | 0.8401 | 39 | 参与的话是一千多然后直播间里面是十 | cer | 0.0556 |
| 320 | causal_full_asr | emilia_zh_0005600573 | 0.8216 | 44 | 参与的话是一千多然后直播间里面是十 | cer | 0.0556 |
| 640 | causal_full_asr | emilia_zh_0005600573 | 0.7918 | 47 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 1280 | causal_full_asr | emilia_zh_0005600573 | 0.8104 | 44 | 参与的话是一千多然后直播间里面是十个人 | cer | 0.0556 |
| 160 | causal_full_asr | emilia_zh_0005713628 | 0.7123 | 61 | 大家会联想到中国的话大家会想到的是什么名山大川 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005713628 | 0.7032 | 61 | 大家会联想到中国的话大家会想到的是什么名山大川 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005713628 | 0.7123 | 59 | 大家会联想到中国化大家会想到的是什么名山大川 | cer | 0.0870 |
| 1280 | causal_full_asr | emilia_zh_0005713628 | 0.7032 | 60 | 大家会联想到中国话大家会想到的是什么名山大川 | cer | 0.0435 |
| 160 | causal_full_asr | emilia_zh_0005714451 | 0.8077 | 51 | 顺便撞到您接下来你可以看到其实前面的三目标 | cer | 0.4400 |
| 320 | causal_full_asr | emilia_zh_0005714451 | 0.8112 | 52 | 生产状态您接下来你可以看到其实前面的三目标 | cer | 0.2800 |
| 640 | causal_full_asr | emilia_zh_0005714451 | 0.7762 | 56 | 生产状态您接下来你可以看到其实前面的三目标 | cer | 0.2800 |
| 1280 | causal_full_asr | emilia_zh_0005714451 | 0.7832 | 56 | 生产状态您接下来你可以看到其实前边的三目标 | cer | 0.2400 |
| 160 | causal_full_asr | emilia_zh_0005780397 | 0.6890 | 162 | 工作对一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小黄然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0462 |
| 320 | causal_full_asr | emilia_zh_0005780397 | 0.6701 | 172 | 工作对一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做hr嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0462 |
| 640 | causal_full_asr | emilia_zh_0005780397 | 0.6873 | 168 | 工作都一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小朋友们然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0308 |
| 1280 | causal_full_asr | emilia_zh_0005780397 | 0.6890 | 168 | 工作都一定是对立的然后工作一定是消耗的所以在前两天呢就是我也做HR嘛然后就跟一些小朋友然后做圆桌讨论的时候也想知道他们对工作怎么看 | cer | 0.0154 |
| 160 | causal_full_asr | emilia_zh_0005852724 | 0.7546 | 59 | 这个算重要的这身为妈妈我一直希望我的两个女儿可以被 | cer | 0.1786 |
| 320 | causal_full_asr | emilia_zh_0005852724 | 0.7361 | 63 | 这个是很重要的所以作为妈妈我一直希望我的两个女儿可以被 | cer | 0.0714 |
| 640 | causal_full_asr | emilia_zh_0005852724 | 0.7212 | 67 | 这个是很重要的所以身为妈妈我一直希望我的两个女儿可以被 | cer | 0.0357 |
| 1280 | causal_full_asr | emilia_zh_0005852724 | 0.7212 | 68 | 这个是很重要的所以身为妈妈我一直希望我的两个女儿可以被 | cer | 0.0357 |
| 160 | causal_full_asr | emilia_zh_0005926417 | 0.7403 | 85 | 没有办法判就我自己没有办法去得出这样的判断和结论出来可能只能是参考一些呃 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005926417 | 0.7403 | 89 | 没有办法判就我自己没有办法去得出这样的判断和结论出来从只能是参考一些呃 | cer | 0.0556 |
| 640 | causal_full_asr | emilia_zh_0005926417 | 0.7320 | 90 | 没有办法判断就我自己没有办法去得出这样的判断或者结论出来横指可能是参考一些呃 | cer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0005926417 | 0.7320 | 92 | 没有办法判就我自己没有办法去得出这样的判断或者结论出来横指可能是参考一些呃 | cer | 0.1389 |
| 160 | causal_full_asr | emilia_zh_0005960185 | 0.7004 | 62 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天主葵 | cer | 0.2083 |
| 320 | causal_full_asr | emilia_zh_0005960185 | 0.7048 | 62 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.1667 |
| 640 | causal_full_asr | emilia_zh_0005960185 | 0.6916 | 63 | 哦我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.1667 |
| 1280 | causal_full_asr | emilia_zh_0005960185 | 0.6872 | 63 | 啊我知道一块但是那发现很多人觉得不是玫瑰是天竺葵 | cer | 0.2083 |
| 160 | causal_full_asr | emilia_zh_0005960324 | 0.7287 | 93 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0005960324 | 0.7313 | 93 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0005960324 | 0.7235 | 96 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0005960324 | 0.7364 | 91 | 他说本来呢我想着你这一趟就不麻烦大家了等我们在北京办典礼的时候再请大家来啊 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0006041799 | 0.6960 | 64 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 320 | causal_full_asr | emilia_zh_0006041799 | 0.6740 | 67 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 640 | causal_full_asr | emilia_zh_0006041799 | 0.6564 | 66 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 1280 | causal_full_asr | emilia_zh_0006041799 | 0.6608 | 66 | 在这个委托发生的十年前两个人就合伙开了这家面馆 | cer | 0.0417 |
| 160 | causal_full_asr | emilia_zh_0006056200 | 0.7553 | 69 | 就是你又还是要用一些就是能看不掉的东西要不然的话你这氛围太大了 | cer | 0.3030 |
| 320 | causal_full_asr | emilia_zh_0006056200 | 0.7311 | 78 | 就你又还是要用一些就是能看不掉的东西要不然的话你这复印太大了 | cer | 0.2727 |
| 640 | causal_full_asr | emilia_zh_0006056200 | 0.7281 | 80 | 就你又还是要有一些就是能看不掉的东西要不然的话你这风险太大了 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0006056200 | 0.7130 | 79 | 就你又还是要有一些就是能看的更多的东西要不然的话你的风险还大了 | cer | 0.2424 |
| 160 | causal_full_asr | emilia_zh_0006099356 | 0.6515 | 75 | 就整个这个流向还比较多但是你非得走你确定是能拿到钱的，是吗 | cer | 0.1852 |
| 320 | causal_full_asr | emilia_zh_0006099356 | 0.6432 | 79 | 就整个这个流量还不比较多但是你非得知道你确定是能拿到钱的什么 | cer | 0.2963 |
| 640 | causal_full_asr | emilia_zh_0006099356 | 0.6473 | 77 | 就整个这个流量还不叫多但是你确定是能拿到钱的，是吗 | cer | 0.2593 |
| 1280 | causal_full_asr | emilia_zh_0006099356 | 0.6473 | 78 | 就整个这个流量还不叫多但是你确定这种你确定是能拿到钱的，是吗 | cer | 0.2222 |
| 160 | causal_full_asr | emilia_zh_0006174175 | 0.7393 | 102 | 我一上来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确知道这个事儿算什么 | cer | 0.2174 |
| 320 | causal_full_asr | emilia_zh_0006174175 | 0.7299 | 107 | 一章后来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确知道这个事儿算什么 | cer | 0.1957 |
| 640 | causal_full_asr | emilia_zh_0006174175 | 0.7180 | 105 | 一章后来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确的知道这个事儿算什么 | cer | 0.1739 |
| 1280 | causal_full_asr | emilia_zh_0006174175 | 0.7085 | 109 | 一扔后来就几个老师在一块偷偷的讨论什么呢大家就是老师都没有人说特别明确的知道这个事儿算什么 | cer | 0.1739 |
| 160 | causal_full_asr | emilia_zh_0006270577 | 0.6884 | 84 | 用直接用三粉盖其实可以的啊这个演部哪里我们并不是所有的人都是需要的 | cer | 0.1818 |
| 320 | causal_full_asr | emilia_zh_0006270577 | 0.6747 | 83 | 用直接用三份盖其实可以的啊这个演部哪里用并不是所有的人都是需要的 | cer | 0.2121 |
| 640 | causal_full_asr | emilia_zh_0006270577 | 0.6849 | 81 | 用直接用三粉盖其实可以的啊这个演部哪里我们并不是所有的人都是需要的 | cer | 0.1818 |
| 1280 | causal_full_asr | emilia_zh_0006270577 | 0.6712 | 83 | 用直接用三粉盖其实可以的啊这个演部哪里有并不是所有的人都是需要的 | cer | 0.1818 |
| 160 | causal_full_asr | emilia_zh_0006330755 | 0.5991 | 66 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 320 | causal_full_asr | emilia_zh_0006330755 | 0.5714 | 63 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0006330755 | 0.5714 | 64 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 1280 | causal_full_asr | emilia_zh_0006330755 | 0.5622 | 65 | An hour later they were standing in the graveyard of the old stone church | wer | 0.1429 |
| 160 | causal_full_asr | emilia_zh_0006404958 | 0.6429 | 74 | The characters are like monkeys in winter second up river wines where drinking water | wer | 0.2667 |
| 320 | causal_full_asr | emilia_zh_0006404958 | 0.6607 | 68 | The characters are like monkeys in winter second up river wines where drinking water | wer | 0.2667 |
| 640 | causal_full_asr | emilia_zh_0006404958 | 0.6929 | 63 | The characters are like monkeys in winter second up river wines where drinking water | wer | 0.2667 |
| 1280 | causal_full_asr | emilia_zh_0006404958 | 0.6750 | 67 | The characters are like monkeys in winter shaking up river wines while drinking water | wer | 0.1333 |
| 160 | causal_full_asr | emilia_zh_0006405437 | 0.7517 | 77 | The two little boys can't have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 320 | causal_full_asr | emilia_zh_0006405437 | 0.7244 | 79 | The two little boys can have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 640 | causal_full_asr | emilia_zh_0006405437 | 0.7130 | 81 | The two little boys can have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 1280 | causal_full_asr | emilia_zh_0006405437 | 0.6925 | 81 | The two little boys can't have club Tax rules are as clear as mud Tax rules are as clear as mud | wer | 0.1500 |
| 160 | causal_full_asr | emilia_zh_0006502797 | 0.6904 | 129 | 我觉得这个委托人可不冷我觉得那你说古代的时候和他的手套那古人都已经挖到比如说这个明代开始就把这核桃把那个发明石头 | cer | 0.3443 |
| 320 | causal_full_asr | emilia_zh_0006502797 | 0.6637 | 135 | 我觉得这个胃腿不认可我觉得那时候古代的时候喝黑的舌头那古人都已经玩了比如说这个明代开始就把这黑的舌头就变成外面石头了 | cer | 0.4262 |
| 640 | causal_full_asr | emilia_zh_0006502797 | 0.6459 | 140 | 我觉得这个胃特别不认可我觉得那时候古代的时候喝他的舌头那古人都已经挖了比如说这个明代开始就把这核桃就变成外面石头了 | cer | 0.3607 |
| 1280 | causal_full_asr | emilia_zh_0006502797 | 0.6437 | 138 | 我觉得这个胃特别不认可我觉得那时候古代的时候喝喝的时候他那古人都已经玩了比如说这个明代开始就把这核桃就发明出套了 | cer | 0.2951 |
| 160 | causal_full_asr | emilia_zh_0006598177 | 0.8000 | 73 | 他说这个女性受害人这些案件当中都百分之五十几都是这个亲密关系 | cer | 0.0968 |
| 320 | causal_full_asr | emilia_zh_0006598177 | 0.7780 | 78 | 而说这个女性受害人这些案件当中有百分之五十级都是这个亲密关系 | cer | 0.0968 |
| 640 | causal_full_asr | emilia_zh_0006598177 | 0.7829 | 80 | 啊说这个女性受害人这些案件当中有百分之五十级都是这个亲密关系 | cer | 0.0645 |
| 1280 | causal_full_asr | emilia_zh_0006598177 | 0.7854 | 80 | 啊说这个女性受害人这些案件当中有百分之啊五十级都是这个亲密关系 | cer | 0.0323 |
| 160 | causal_full_asr | emilia_zh_0006610442 | 0.7126 | 39 | 我觉得这这这有什么有意思就是这种 | cer | 0.2778 |
| 320 | causal_full_asr | emilia_zh_0006610442 | 0.7365 | 39 | 我觉得这这这什么有意思就是这种 | cer | 0.2778 |
| 640 | causal_full_asr | emilia_zh_0006610442 | 0.6886 | 45 | 我觉得这这这也是蛮有意思就是这种 | cer | 0.1111 |
| 1280 | causal_full_asr | emilia_zh_0006610442 | 0.7006 | 43 | 我觉得这这这也是蛮有意思就是这种 | cer | 0.1111 |
| 160 | causal_full_asr | emilia_zh_0006659550 | 0.7352 | 58 | 中期的数据看其有利于这个布什让他能做出两次错误的判断 | cer | 0.2414 |
| 320 | causal_full_asr | emilia_zh_0006659550 | 0.7273 | 61 | 中期的数据看上去有利于这个布什但他能做出两次错误的判断 | cer | 0.1724 |
| 640 | causal_full_asr | emilia_zh_0006659550 | 0.7154 | 67 | 中期的数据看起来有利这个不实那他们做出的两次错误的判断 | cer | 0.2069 |
| 1280 | causal_full_asr | emilia_zh_0006659550 | 0.7115 | 67 | 中期的数据看上去有利于这个布什但他能做出两次错误的判断 | cer | 0.1724 |
| 160 | causal_full_asr | emilia_zh_0006884293 | 0.7543 | 93 | 我不明白这张他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0571 |
| 320 | causal_full_asr | emilia_zh_0006884293 | 0.7420 | 96 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0006884293 | 0.7322 | 96 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0006884293 | 0.7346 | 97 | 我不明白支撑他继续下去的动力是什么难道是人类所说的那种虚无缥缈的感情吗 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0006992873 | 0.8158 | 36 | 绿色的城堡稻草人首先警觉起来 | cer | 0.1429 |
| 320 | causal_full_asr | emilia_zh_0006992873 | 0.8158 | 34 | 绿色的城堡稻草人首先警觉起来 | cer | 0.1429 |
| 640 | causal_full_asr | emilia_zh_0006992873 | 0.8026 | 39 | 绿色的城堡稻草人首先警叫起来 | cer | 0.0714 |
| 1280 | causal_full_asr | emilia_zh_0006992873 | 0.8114 | 36 | 绿色的城堡稻草人首先警叫起来 | cer | 0.0714 |
| 160 | causal_full_asr | emilia_zh_0007060544 | 0.7250 | 40 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 320 | causal_full_asr | emilia_zh_0007060544 | 0.7063 | 43 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 640 | causal_full_asr | emilia_zh_0007060544 | 0.7063 | 42 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0007060544 | 0.7125 | 41 | 如果你做的事情真正对别人有价值 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0007353988 | 0.7675 | 111 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是有耳的侦察车 | cer | 0.0638 |
| 320 | causal_full_asr | emilia_zh_0007353988 | 0.7509 | 123 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是有耳的侦察车 | cer | 0.0638 |
| 640 | causal_full_asr | emilia_zh_0007353988 | 0.7417 | 129 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0000 |
| 1280 | causal_full_asr | emilia_zh_0007353988 | 0.7454 | 127 | 两颗核弹倒是攻击了驱逐者的活动区域但第一颗被能量防护区域偏转第二颗打中了一艘也许是诱饵的侦察船 | cer | 0.0000 |
| 160 | causal_full_asr | emilia_zh_0007761003 | 0.7770 | 52 | 如果你希望你的团队成员陪你走得更远那你就需要 | cer | 0.0455 |
| 320 | causal_full_asr | emilia_zh_0007761003 | 0.7732 | 49 | 如果你向你的团队成员分析走得更远那你就需要 | cer | 0.1818 |
| 640 | causal_full_asr | emilia_zh_0007761003 | 0.7621 | 55 | 如果你希望你的团队成员陪你走得更远那你就需要 | cer | 0.0455 |
| 1280 | causal_full_asr | emilia_zh_0007761003 | 0.7621 | 55 | 如果你希望你的团队成员陪你走得更远那你就需要 | cer | 0.0455 |
| 160 | streaming_asr | CommonVoice_EN_0000042263 | 0.7526 | 35 | There remain a the year was spent in court | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000042263 | 0.7211 | 36 | There remain of the year was spent in port | wer | 0.2222 |
| 640 | streaming_asr | CommonVoice_EN_0000042263 | 0.6947 | 39 | That remained of the year was spent in port | wer | 0.2222 |
| 1280 | streaming_asr | CommonVoice_EN_0000042263 | 0.7053 | 38 | There remain of the year was spent in port | wer | 0.2222 |
| 160 | streaming_asr | CommonVoice_EN_0000042530 | 0.7204 | 84 | The international exchange center is a multiplefunctional building with capturia accommodation and classrooms | wer | 0.1538 |
| 320 | streaming_asr | CommonVoice_EN_0000042530 | 0.6872 | 94 | The international exchange center is in multiplefunctional building with capitalia accommodation and classrooms | wer | 0.2308 |
| 640 | streaming_asr | CommonVoice_EN_0000042530 | 0.6517 | 98 | The international exchange center is in multiplefunctional building with capitalia accommodation and classrooms | wer | 0.2308 |
| 1280 | streaming_asr | CommonVoice_EN_0000042530 | 0.6540 | 97 | The international exchange center is in multiplefunctional building with capt interior accommodation and classrooms | wer | 0.3077 |
| 160 | streaming_asr | CommonVoice_EN_0000068601 | 0.7665 | 71 | The archival of Liberalism collects document on the history of allevated Liberalism | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000068601 | 0.7500 | 75 | The archipelago of Liberalism collects document on the history of allevized Oberilism | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000068601 | 0.7429 | 76 | The archival of Liberalism collects document on the history of allevized Oberilese | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000068601 | 0.7217 | 77 | The archival the freemeworthy collects document on the history of allevized Olympias | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000116742 | 0.7951 | 35 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000116742 | 0.7541 | 38 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000116742 | 0.7705 | 34 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000116742 | 0.7746 | 36 | It is one after better now mountains in Sweden | wer | 0.3333 |
| 160 | streaming_asr | CommonVoice_EN_0000126367 | 0.7179 | 63 | He had always said grids of retired for exicism and charity to the poor | wer | 0.3846 |
| 320 | streaming_asr | CommonVoice_EN_0000126367 | 0.6750 | 65 | He had always a grudge of preterious exicism and charity to the poor | wer | 0.3077 |
| 640 | streaming_asr | CommonVoice_EN_0000126367 | 0.6714 | 67 | He had always a grudge of preterious exicism and charity to the poor | wer | 0.3077 |
| 1280 | streaming_asr | CommonVoice_EN_0000126367 | 0.6714 | 68 | He had always to grew up before exicism and charity to the poor | wer | 0.3846 |
| 160 | streaming_asr | CommonVoice_EN_0000159911 | 0.7711 | 44 | The series resears particular claimed during the vamire storylines | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000159911 | 0.7430 | 49 | The serious rec seizure particular claim during the vampire storylines | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000159911 | 0.7430 | 50 | The serious recused particular clame during the vampire storylines | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000159911 | 0.7148 | 54 | The serious re seizure particular claim during the the vampire storylines | wer | 0.5556 |
| 160 | streaming_asr | CommonVoice_EN_0000188343 | 0.8504 | 16 | They have such as thoroughly | wer | 0.4000 |
| 320 | streaming_asr | CommonVoice_EN_0000188343 | 0.8268 | 19 | They have such as thoroughly | wer | 0.4000 |
| 640 | streaming_asr | CommonVoice_EN_0000188343 | 0.8425 | 17 | They have such as thoroughly the | wer | 0.6000 |
| 1280 | streaming_asr | CommonVoice_EN_0000188343 | 0.8031 | 18 | Day of such as thoroughly the | wer | 1.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000189191 | 0.8763 | 17 | The boy brought his horse closer | wer | 0.0000 |
| 320 | streaming_asr | CommonVoice_EN_0000189191 | 0.8441 | 22 | The boy brought his whorse closser | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000189191 | 0.8118 | 23 | The boy brought his whorse closser | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000189191 | 0.7957 | 24 | The boy brought his wharst closer | wer | 0.1667 |
| 160 | streaming_asr | CommonVoice_EN_0000237791 | 0.7650 | 32 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 320 | streaming_asr | CommonVoice_EN_0000237791 | 0.7486 | 33 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 640 | streaming_asr | CommonVoice_EN_0000237791 | 0.7322 | 35 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 1280 | streaming_asr | CommonVoice_EN_0000237791 | 0.7377 | 34 | The dusty bench stood by its stone wall | wer | 0.1250 |
| 160 | streaming_asr | CommonVoice_EN_0000238037 | 0.9318 | 17 | Who find a fire in that book You are the right Which book | wer | 0.9286 |
| 320 | streaming_asr | CommonVoice_EN_0000238037 | 0.9221 | 19 | Who find a fire in now the moon for the white which you | wer | 0.8571 |
| 640 | streaming_asr | CommonVoice_EN_0000238037 | 0.8929 | 25 | New find of the fire in Now we need to preside the book on | wer | 0.7857 |
| 1280 | streaming_asr | CommonVoice_EN_0000238037 | 0.9156 | 23 | You find the fire in now before you start spire the pick on | wer | 0.8571 |
| 160 | streaming_asr | CommonVoice_EN_0000263325 | 0.7386 | 41 | He has one younger brother Trevor read Nelson | wer | 0.1250 |
| 320 | streaming_asr | CommonVoice_EN_0000263325 | 0.7303 | 45 | He has one younger brother travour read Nelson | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000263325 | 0.7303 | 42 | He has one younger brother traveller read Nelson | wer | 0.2500 |
| 1280 | streaming_asr | CommonVoice_EN_0000263325 | 0.7095 | 44 | He has one younger brother traveller read Nelson | wer | 0.2500 |
| 160 | streaming_asr | CommonVoice_EN_0000285524 | 0.7635 | 60 | The townships proximity to the sea favored marine trade and fish factors | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000285524 | 0.7407 | 64 | The townships pro proximity to the sea favored marine trade and fish factors | wer | 0.3333 |
| 640 | streaming_asr | CommonVoice_EN_0000285524 | 0.7151 | 67 | The townships pro proximity to the sea favored marine trade and fish f factories | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000285524 | 0.7179 | 67 | The townships pro proximity to the sea favored moring tried in fish factories | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000286138 | 0.7316 | 57 | The boy reminded the old man that he had said something about hidden dresses | wer | 0.0714 |
| 320 | streaming_asr | CommonVoice_EN_0000286138 | 0.6901 | 65 | The boy reminded the old man that he had said something about hidden dresses | wer | 0.0714 |
| 640 | streaming_asr | CommonVoice_EN_0000286138 | 0.6837 | 66 | The boy reminded the old man that he had said something about hidden treasure | wer | 0.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000286138 | 0.6581 | 70 | The boy reminded the old man that he had said something about hidden treasure | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000311292 | 0.7064 | 50 | In head many exceptional qualities compared to previous aircraft | wer | 0.2222 |
| 320 | streaming_asr | CommonVoice_EN_0000311292 | 0.6723 | 53 | In head many exceptional qualities compared to previous aircraft | wer | 0.2222 |
| 640 | streaming_asr | CommonVoice_EN_0000311292 | 0.6809 | 52 | It had many exceptional qualities compared to previous aircraft | wer | 0.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000311292 | 0.6766 | 57 | It had many exceptional qualities compared to previous aircraft | wer | 0.0000 |
| 160 | streaming_asr | CommonVoice_EN_0000331841 | 0.7692 | 23 | You did roll the some of last time | wer | 0.6250 |
| 320 | streaming_asr | CommonVoice_EN_0000331841 | 0.7538 | 22 | You did wrote the some of last time | wer | 0.6250 |
| 640 | streaming_asr | CommonVoice_EN_0000331841 | 0.7462 | 24 | You did wrote the some of last time | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000331841 | 0.7385 | 26 | You did wrote the some of last time | wer | 0.6250 |
| 160 | streaming_asr | CommonVoice_EN_0000332324 | 0.7565 | 38 | Many building post for a wood where entirely destroyed | wer | 0.5556 |
| 320 | streaming_asr | CommonVoice_EN_0000332324 | 0.7304 | 43 | Many building both for a wide war entirely destroyed | wer | 0.5556 |
| 640 | streaming_asr | CommonVoice_EN_0000332324 | 0.7043 | 46 | Many building both for a wood war entirely destroyed | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000332324 | 0.6957 | 47 | Many building both for a wood war entirely destroyed | wer | 0.4444 |
| 160 | streaming_asr | CommonVoice_EN_0000352398 | 0.8497 | 35 | He confided his high school from a ran lull all high school | wer | 0.4545 |
| 320 | streaming_asr | CommonVoice_EN_0000352398 | 0.8235 | 38 | He complicated his high school form of ran L all high school | wer | 0.5455 |
| 640 | streaming_asr | CommonVoice_EN_0000352398 | 0.7974 | 41 | He completed his high school from a ram lull all high school | wer | 0.2727 |
| 1280 | streaming_asr | CommonVoice_EN_0000352398 | 0.8301 | 35 | He completed his high school from a random life all high school | wer | 0.3636 |
| 160 | streaming_asr | CommonVoice_EN_0000381265 | 0.7829 | 26 | You existed in playing megan | wer | 0.8000 |
| 320 | streaming_asr | CommonVoice_EN_0000381265 | 0.7632 | 27 | You existed in playing megan | wer | 0.8000 |
| 640 | streaming_asr | CommonVoice_EN_0000381265 | 0.7763 | 27 | You vexisted in playing megan | wer | 1.0000 |
| 1280 | streaming_asr | CommonVoice_EN_0000381265 | 0.7566 | 29 | You've existed since playing megan | wer | 0.6000 |
| 160 | streaming_asr | CommonVoice_EN_0000381462 | 0.7802 | 35 | Chirrod spent the rest of his little life with Mrusus | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000381462 | 0.7629 | 38 | Chirrod spent a rest of his slow life with Mrusus | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000381462 | 0.7457 | 42 | Chirrived spent the rest of his slow life would Mrusus | wer | 0.4444 |
| 1280 | streaming_asr | CommonVoice_EN_0000381462 | 0.7457 | 43 | Girard spent the rest of his slept life with Mrusus | wer | 0.2222 |
| 160 | streaming_asr | CommonVoice_EN_0000430128 | 0.7520 | 45 | Lord talk in your retains his captancy with designed white rose bad | wer | 0.6429 |
| 320 | streaming_asr | CommonVoice_EN_0000430128 | 0.7400 | 49 | Lu Tuo can you hear his captancy designed their white rows bad | wer | 0.7857 |
| 640 | streaming_asr | CommonVoice_EN_0000430128 | 0.7280 | 54 | Lord Hawke can ear daily's his captancy designed their white rows bad | wer | 0.6429 |
| 1280 | streaming_asr | CommonVoice_EN_0000430128 | 0.7200 | 54 | Lord Hawke can ear daily's his captancy designed their white rows bad | wer | 0.6429 |
| 160 | streaming_asr | CommonVoice_EN_0000433992 | 0.7159 | 56 | Nobody wants to the discuss how you'll all end up trap the at kind lunch box | wer | 0.7143 |
| 320 | streaming_asr | CommonVoice_EN_0000433992 | 0.6937 | 56 | Nobody wants to the discuss how you'll all end up trap and I can't learn box | wer | 0.7143 |
| 640 | streaming_asr | CommonVoice_EN_0000433992 | 0.6605 | 63 | Nobody wants to the discuss how you all end up trap and a kind lunch box | wer | 0.6429 |
| 1280 | streaming_asr | CommonVoice_EN_0000433992 | 0.6605 | 60 | Nobody wanted to discuss how you all ended up trap and a charm lunch box | wer | 0.4286 |
| 160 | streaming_asr | CommonVoice_EN_0000434002 | 0.8099 | 17 | He idea frightened him | wer | 0.2500 |
| 320 | streaming_asr | CommonVoice_EN_0000434002 | 0.7686 | 19 | He idea frightened him | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000434002 | 0.7603 | 19 | He idea frightened him | wer | 0.2500 |
| 1280 | streaming_asr | CommonVoice_EN_0000434002 | 0.7686 | 19 | He idea frightened him | wer | 0.2500 |
| 160 | streaming_asr | CommonVoice_EN_0000471209 | 0.7603 | 29 | The boy began to do dear into the journ | wer | 0.3750 |
| 320 | streaming_asr | CommonVoice_EN_0000471209 | 0.6849 | 34 | The boy began to deag into the journ | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000471209 | 0.6644 | 36 | The boy began to do dear and to the june | wer | 0.6250 |
| 1280 | streaming_asr | CommonVoice_EN_0000471209 | 0.6781 | 36 | The boy began to do dear and to the june | wer | 0.6250 |
| 160 | streaming_asr | CommonVoice_EN_0000501889 | 0.6684 | 41 | We usually part of motors cycles near to this building | wer | 0.4444 |
| 320 | streaming_asr | CommonVoice_EN_0000501889 | 0.6218 | 43 | We usually part of motors cycles near to this building | wer | 0.4444 |
| 640 | streaming_asr | CommonVoice_EN_0000501889 | 0.6218 | 43 | We usually part of motocycles near to this building | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000501889 | 0.6580 | 41 | We usually part of model cycles near to this building | wer | 0.4444 |
| 160 | streaming_asr | CommonVoice_EN_0000519794 | 0.6724 | 54 | To the press of unecrization is unless the then is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.6667 |
| 320 | streaming_asr | CommonVoice_EN_0000519794 | 0.6638 | 60 | Through the prize of unecrization is unless the dend is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.6667 |
| 640 | streaming_asr | CommonVoice_EN_0000519794 | 0.6724 | 54 | Give the prize of unacquisition is unless the then is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.6667 |
| 1280 | streaming_asr | CommonVoice_EN_0000519794 | 0.6595 | 56 | You've prized a in acquisition is unless the then is under<\|glm_semantic_774\|><\|write_generate\|><\|eng\|><\|start_content\|>under | wer | 0.7500 |
| 160 | streaming_asr | CommonVoice_EN_0000520146 | 0.6793 | 42 | Manage trucks or supported by a leave spring suspensions | wer | 0.5000 |
| 320 | streaming_asr | CommonVoice_EN_0000520146 | 0.6576 | 44 | Many trucks are supported by a leave spring suspensions | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000520146 | 0.6522 | 44 | Manage trucks or supported by a leave spring suspensions | wer | 0.5000 |
| 1280 | streaming_asr | CommonVoice_EN_0000520146 | 0.6359 | 44 | Manage trucks or supported by a leave spring suspensions | wer | 0.5000 |
| 160 | streaming_asr | CommonVoice_EN_0000555807 | 0.8491 | 27 | The color intensifies as a stop brightens | wer | 0.2857 |
| 320 | streaming_asr | CommonVoice_EN_0000555807 | 0.8208 | 30 | The color intensifies as a star brightens | wer | 0.1429 |
| 640 | streaming_asr | CommonVoice_EN_0000555807 | 0.8019 | 33 | The color intensifies as a star brightens | wer | 0.1429 |
| 1280 | streaming_asr | CommonVoice_EN_0000555807 | 0.7972 | 34 | The color intensifies as a star brightens | wer | 0.1429 |
| 160 | streaming_asr | CommonVoice_EN_0000555853 | 0.7869 | 40 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 320 | streaming_asr | CommonVoice_EN_0000555853 | 0.7582 | 40 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 640 | streaming_asr | CommonVoice_EN_0000555853 | 0.7336 | 41 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 1280 | streaming_asr | CommonVoice_EN_0000555853 | 0.7418 | 42 | Shaving water Sam he said from within the cottons | wer | 0.3750 |
| 160 | streaming_asr | CommonVoice_EN_0000593898 | 0.6964 | 63 | In recent years Saint George's has spent the hundredth Medical School roots | wer | 0.3333 |
| 320 | streaming_asr | CommonVoice_EN_0000593898 | 0.6893 | 59 | In recent years Saint George's has expended by its medical school roots | wer | 0.2500 |
| 640 | streaming_asr | CommonVoice_EN_0000593898 | 0.6714 | 61 | In recent years St George's has a span beyond its medical school roots | wer | 0.3333 |
| 1280 | streaming_asr | CommonVoice_EN_0000593898 | 0.6750 | 63 | In recently years St George's has spent beyond its medical school roots | wer | 0.3333 |
| 160 | streaming_asr | DailyTalk_0000001997 | 0.8352 | 11 | In Miss Scott please | wer | 0.7500 |
| 320 | streaming_asr | DailyTalk_0000001997 | 0.7802 | 15 | In Miss Scott Please | wer | 0.7500 |
| 640 | streaming_asr | DailyTalk_0000001997 | 0.7582 | 15 | Give me Scott Please | wer | 0.2500 |
| 1280 | streaming_asr | DailyTalk_0000001997 | 0.7692 | 14 | Give me Scott Please | wer | 0.2500 |
| 160 | streaming_asr | EN_B00013_S05834_W000745 | 0.5632 | 105 | Even if the night could still stand through a strong resilience the broken bones and is chested with making him lose any ability to fight | wer | 0.2800 |
| 320 | streaming_asr | EN_B00013_S05834_W000745 | 0.5431 | 110 | Even if the night could still stand through a strong resilience the broken bones and a chest would make him lose any ability to fight | wer | 0.1600 |
| 640 | streaming_asr | EN_B00013_S05834_W000745 | 0.5517 | 107 | Even if the night could still stand through a strong resilience the broken bones and a chest would make him lose any ability to fight | wer | 0.1600 |
| 1280 | streaming_asr | EN_B00013_S05834_W000745 | 0.5374 | 109 | Even if the night could still stand through a strong resilience the broken bones and his chest would make him lose any ability to fight | wer | 0.1200 |
| 160 | streaming_asr | EN_B00013_S05888_W000041 | 0.6952 | 142 | But I've mean I have a big friend I love I don't love to see each other and absession I've been big friend chad's fort ages and say he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3878 |
| 320 | streaming_asr | EN_B00013_S05888_W000041 | 0.6771 | 148 | That I've mean I'm a big person I love I don't love see each other and absession I've been big friend chad's fort ages and been saying he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3265 |
| 640 | streaming_asr | EN_B00013_S05888_W000041 | 0.6540 | 155 | The I've mean I'm a big child I love I don't love to see each other opsession I've been big friend chad's four ages and been saying he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3061 |
| 1280 | streaming_asr | EN_B00013_S05888_W000041 | 0.6639 | 148 | The I've mean I'm a big child I love I I love to see each other opinion I've been big friend chad's four ages and been saying he needs more making him the guy He could be the guy He could be the next COVID | wer | 0.3061 |
| 160 | streaming_asr | EN_B00013_S06799_W000009 | 0.6484 | 106 | Now with that we can have better model is safety fairness and help any users to have better resibility appropriate trust | wer | 0.2381 |
| 320 | streaming_asr | EN_B00013_S06799_W000009 | 0.6256 | 108 | Now with that we can have better model and sure safety fairness and help any user to have better resibility appropriate trust | wer | 0.2381 |
| 640 | streaming_asr | EN_B00013_S06799_W000009 | 0.6256 | 106 | And with that we can have better model and sure 50 fairness and help and user to have better visibility appropriate trust | wer | 0.2381 |
| 1280 | streaming_asr | EN_B00013_S06799_W000009 | 0.6324 | 105 | And when that we can have better model and sure 50 furnace and help and user to have better visibility appropriate trust | wer | 0.3333 |
| 160 | streaming_asr | EN_B00036_S05316_W000048 | 0.7162 | 57 | This sunderflow sees star has a three foot wide arms span as a taste for sear chance | wer | 0.3529 |
| 320 | streaming_asr | EN_B00036_S05316_W000048 | 0.6766 | 67 | This sunflower sees star has a three foot wide arms span and taste for sear chance | wer | 0.2941 |
| 640 | streaming_asr | EN_B00036_S05316_W000048 | 0.6502 | 65 | This sunflower sees star has a three foot wide arms span at the taste for sear chance | wer | 0.3529 |
| 1280 | streaming_asr | EN_B00036_S05316_W000048 | 0.6172 | 69 | This sunflower sees star has a three foot wide arms span as the taste for sear chance | wer | 0.3529 |
| 160 | streaming_asr | EN_B00043_S01954_W000033 | 0.6746 | 111 | Because love is more than just in emotion It's a capacity of verb and enlessly renewable resource And not just in our private life is | wer | 0.2500 |
| 320 | streaming_asr | EN_B00043_S01954_W000033 | 0.6509 | 111 | Because love is more than just in emotion It's a capacity of verb and enlessly renewable resource And not just in our private life is | wer | 0.2500 |
| 640 | streaming_asr | EN_B00043_S01954_W000033 | 0.6358 | 111 | Because love is more than just in emotion It's a capacity of verb and enlessly renovable resource And not just in our private life is | wer | 0.2917 |
| 1280 | streaming_asr | EN_B00043_S01954_W000033 | 0.6315 | 114 | Because love is more than just in emotion It's a capacity of verb and enlessly renewable resource And not just in our private life | wer | 0.2083 |
| 160 | streaming_asr | EN_B00043_S02661_W000018 | 0.7283 | 48 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 320 | streaming_asr | EN_B00043_S02661_W000018 | 0.7208 | 52 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 640 | streaming_asr | EN_B00043_S02661_W000018 | 0.6943 | 55 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00043_S02661_W000018 | 0.6755 | 57 | Because opers frequently are medaldreamatic not to say unrealistic | wer | 0.2222 |
| 160 | streaming_asr | EN_B00048_S01234_W000037 | 0.5818 | 46 | And in our case we're not sure this animal is going to be a dog | wer | 0.0625 |
| 320 | streaming_asr | EN_B00048_S01234_W000037 | 0.5212 | 54 | And in our case we're not sure if the animal is going to be a dog | wer | 0.0625 |
| 640 | streaming_asr | EN_B00048_S01234_W000037 | 0.4970 | 57 | And in our case we're not sure if this animal is going to be a dog | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00048_S01234_W000037 | 0.5030 | 59 | And in our case we're not sure if this animal is going to be a dog | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S02289_W000002 | 0.7821 | 38 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 320 | streaming_asr | EN_B00048_S02289_W000002 | 0.7786 | 46 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 640 | streaming_asr | EN_B00048_S02289_W000002 | 0.7750 | 44 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 1280 | streaming_asr | EN_B00048_S02289_W000002 | 0.7714 | 44 | This kind of muscle is mostly collected to my bones | wer | 0.1000 |
| 160 | streaming_asr | EN_B00048_S02307_W000043 | 0.8223 | 66 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 320 | streaming_asr | EN_B00048_S02307_W000043 | 0.8278 | 59 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 640 | streaming_asr | EN_B00048_S02307_W000043 | 0.8242 | 61 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 1280 | streaming_asr | EN_B00048_S02307_W000043 | 0.8168 | 65 | Let's check He likes to eat Peter I don't want to eat this salad That isn't my shake | wer | 0.0556 |
| 160 | streaming_asr | EN_B00048_S03599_W000339 | 0.7656 | 49 | That's turned around I've my speak froze the blood of everyone close by | wer | 0.2308 |
| 320 | streaming_asr | EN_B00048_S03599_W000339 | 0.7500 | 56 | I turned around I might sreak froze the blood of everyone close by | wer | 0.2308 |
| 640 | streaming_asr | EN_B00048_S03599_W000339 | 0.7438 | 56 | I turned around I and my sreak froze the blood of everyone close by | wer | 0.1538 |
| 1280 | streaming_asr | EN_B00048_S03599_W000339 | 0.7562 | 52 | I turned around and my s shriek froze the blood of everyone close by | wer | 0.0769 |
| 160 | streaming_asr | EN_B00048_S05933_W000060 | 0.5500 | 54 | South America has really interesting culture that would fit | wer | 0.1000 |
| 320 | streaming_asr | EN_B00048_S05933_W000060 | 0.5333 | 55 | South America has a really interesting culture that would fit | wer | 0.0000 |
| 640 | streaming_asr | EN_B00048_S05933_W000060 | 0.5056 | 57 | South America has a really interesting culture that would fit | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00048_S05933_W000060 | 0.4722 | 59 | South American has a really interesting culture that would fit | wer | 0.1000 |
| 160 | streaming_asr | EN_B00048_S05961_W000019 | 0.6369 | 87 | Alogist now use this theory to explain origin of vast of variety of Eukaryotic organisms | wer | 0.2500 |
| 320 | streaming_asr | EN_B00048_S05961_W000019 | 0.6196 | 90 | Biologists now use this theory to explain origin of the vast of arrity of Eukaryotic organisms | wer | 0.1875 |
| 640 | streaming_asr | EN_B00048_S05961_W000019 | 0.6167 | 94 | Biologists now use this theory to explain origin of the vast of arrity of Eukaryotic organisms | wer | 0.1875 |
| 1280 | streaming_asr | EN_B00048_S05961_W000019 | 0.6023 | 94 | Biologists now use this theory to explain origin of the vast of arrity of Eukaryotic organisms | wer | 0.1875 |
| 160 | streaming_asr | EN_B00048_S07042_W000076 | 0.6012 | 49 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 320 | streaming_asr | EN_B00048_S07042_W000076 | 0.5337 | 54 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 640 | streaming_asr | EN_B00048_S07042_W000076 | 0.5337 | 52 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00048_S07042_W000076 | 0.5460 | 53 | I know I didn't have time to put things away before you got here | wer | 0.0000 |
| 160 | streaming_asr | EN_B00048_S07862_W000265 | 0.6995 | 77 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 320 | streaming_asr | EN_B00048_S07862_W000265 | 0.6755 | 79 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 640 | streaming_asr | EN_B00048_S07862_W000265 | 0.6809 | 78 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 1280 | streaming_asr | EN_B00048_S07862_W000265 | 0.6862 | 72 | The only means of crossing large areas of water was in a sailing ship driven by the the wind | wer | 0.0556 |
| 160 | streaming_asr | EN_B00048_S08821_W000040 | 0.6080 | 80 | Are you in participating in a fun raise of this weekend a little like to borrow the van If possible | wer | 0.4444 |
| 320 | streaming_asr | EN_B00048_S08821_W000040 | 0.5914 | 84 | Our union is participating in a fun raise of this weekend and would like to borrow the van if possible | wer | 0.2222 |
| 640 | streaming_asr | EN_B00048_S08821_W000040 | 0.5847 | 81 | Our union is participating in a fun raise of this weekend and would like to borrow the van if possible | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00048_S08821_W000040 | 0.5914 | 82 | Our union is participating in a fun raise of this weekend and would like to borrow the van if possible | wer | 0.2222 |
| 160 | streaming_asr | EN_B00048_S09662_W000003 | 0.6054 | 140 | Okay very good So what don't we need listen to the dialogue for the first time that's listen how President Isaac Holmes says goodbye And then we'll come back and look at the words | wer | 0.1562 |
| 320 | streaming_asr | EN_B00048_S09662_W000003 | 0.5888 | 144 | Okay very good So what don't we need listen to the dialogue for the first time that's listen how President Isaac Holmes has goodbye And then we'll come back and look at the words | wer | 0.1875 |
| 640 | streaming_asr | EN_B00048_S09662_W000003 | 0.5971 | 143 | Okay very good So why don't we let listen to the dialogue for the first time let's listen how President Isaac Holmes has goodbye And then we'll come back and look at the words | wer | 0.1250 |
| 1280 | streaming_asr | EN_B00048_S09662_W000003 | 0.5971 | 144 | Okay very good So why don't we let listen to the dialogue for the first time let's listen how President Isaac Holmes says goodbye And then we'll come back and look at the words | wer | 0.0938 |
| 160 | streaming_asr | EN_B00052_S08802_W000006 | 0.9274 | 15 | Word beautiful Pretty | wer | 0.3333 |
| 320 | streaming_asr | EN_B00052_S08802_W000006 | 0.9085 | 16 | Word beautiful pretty | wer | 0.3333 |
| 640 | streaming_asr | EN_B00052_S08802_W000006 | 0.9117 | 19 | Word beautiful pretty | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00052_S08802_W000006 | 0.9148 | 18 | Word beautiful pretty | wer | 0.3333 |
| 160 | streaming_asr | EN_B00058_S01128_W000118 | 0.6135 | 45 | As a tough foreign or in China I stick out like a sort them | wer | 0.3846 |
| 320 | streaming_asr | EN_B00058_S01128_W000118 | 0.6012 | 47 | As a tall foreign or in China I stick out like a sort them | wer | 0.3077 |
| 640 | streaming_asr | EN_B00058_S01128_W000118 | 0.5828 | 46 | As a tall foreign and China I stick out like a sort them | wer | 0.3077 |
| 1280 | streaming_asr | EN_B00058_S01128_W000118 | 0.5828 | 45 | As a tall foreign and China I stick out like a sort them | wer | 0.3077 |
| 160 | streaming_asr | EN_B00058_S03125_W000010 | 0.6509 | 171 | When tremor comes in contact with a base it changes a good color from yellow to red indicating that suppose solution is a base That is why attered stained turns red when a comes in contact with any kind of base | wer | 0.2143 |
| 320 | streaming_asr | EN_B00058_S03125_W000010 | 0.6193 | 178 | When tremor comes in contact with a base it changes a good color from yellow to red indicating that sopy solution is a base That is what why attered stained turns red when it comes in contact with any kind of base | wer | 0.2143 |
| 640 | streaming_asr | EN_B00058_S03125_W000010 | 0.6066 | 182 | When tremor comes in contact with a base it changes a good color from yellow to red indicating the to soapy solutions is a base That is what why attered stained turns red when it comes in contact with any kind of base | wer | 0.2381 |
| 1280 | streaming_asr | EN_B00058_S03125_W000010 | 0.5893 | 186 | When tremor comes in contact with a base it changes in its color from yellow to red indicating that to soapy solution is a base That is what why attered stained turns red when a come in contact with any kind of base | wer | 0.2143 |
| 160 | streaming_asr | EN_B00058_S03144_W000037 | 0.7143 | 44 | Long complex sentences with multiple paragraphs in email | wer | 0.1111 |
| 320 | streaming_asr | EN_B00058_S03144_W000037 | 0.6741 | 46 | Long complex sentences with multiple paragraphs and email | wer | 0.2222 |
| 640 | streaming_asr | EN_B00058_S03144_W000037 | 0.6786 | 45 | Long complex sentences with multiple paragraphs and email | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00058_S03144_W000037 | 0.6786 | 46 | Long complex sentences with multiple paragraphs and email | wer | 0.2222 |
| 160 | streaming_asr | EN_B00058_S03808_W000018 | 0.6913 | 177 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 320 | streaming_asr | EN_B00058_S03808_W000018 | 0.6675 | 185 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 640 | streaming_asr | EN_B00058_S03808_W000018 | 0.6544 | 189 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 1280 | streaming_asr | EN_B00058_S03808_W000018 | 0.6544 | 190 | We can also say I had gone having two infections in their two indicate a past perfect as well as the perfect tense with the third form So as you can see what inflation does is it changes It actually changes the word | wer | 0.0930 |
| 160 | streaming_asr | EN_B00058_S03815_W000004 | 0.7830 | 54 | My child no her but magic strong enough to make one forget their past | wer | 0.2000 |
| 320 | streaming_asr | EN_B00058_S03815_W000004 | 0.7390 | 60 | My child no her will magic strong enough to make one forget their past | wer | 0.2000 |
| 640 | streaming_asr | EN_B00058_S03815_W000004 | 0.7155 | 61 | My child no her will magic strong enough to make one forget their past | wer | 0.2000 |
| 1280 | streaming_asr | EN_B00058_S03815_W000004 | 0.7185 | 60 | My child no her will magic strong enough to make one forget their past | wer | 0.2000 |
| 160 | streaming_asr | EN_B00058_S04429_W000016 | 0.8119 | 49 | The do see these little green leaves And this big flower will I've a second louvre | wer | 0.3750 |
| 320 | streaming_asr | EN_B00058_S04429_W000016 | 0.7921 | 53 | Do you see these little green luses And this big flower where I've a second louvre | wer | 0.3125 |
| 640 | streaming_asr | EN_B00058_S04429_W000016 | 0.7698 | 58 | Do you see these little green leaves And this being flower where I've a second | wer | 0.2500 |
| 1280 | streaming_asr | EN_B00058_S04429_W000016 | 0.7624 | 61 | Do you see these little green leaves And this being flower where I've a second | wer | 0.2500 |
| 160 | streaming_asr | EN_B00058_S04431_W000023 | 0.6957 | 36 | Use blood head kept all parts of his body warm | wer | 0.2000 |
| 320 | streaming_asr | EN_B00058_S04431_W000023 | 0.6359 | 43 | His blood had kept all parts of his body warm | wer | 0.0000 |
| 640 | streaming_asr | EN_B00058_S04431_W000023 | 0.6630 | 36 | His blood had kept all parts of his body warm | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00058_S04431_W000023 | 0.6467 | 38 | His blood had kept all parts of his body warm | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S06165_W000019 | 0.6582 | 196 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |
| 320 | streaming_asr | EN_B00058_S06165_W000019 | 0.6509 | 197 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |
| 640 | streaming_asr | EN_B00058_S06165_W000019 | 0.6350 | 201 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |
| 1280 | streaming_asr | EN_B00058_S06165_W000019 | 0.6241 | 205 | Or you use the short form where you add a symicolon after the constructor and then you use this question text and now the first argument which is passed a question constructor will be 存储 in the question text property | wer | 0.0976 |
| 160 | streaming_asr | EN_B00058_S06429_W000060 | 0.6020 | 92 | The aliance believed that parsh dis closure be very gigical to do with all of the data that was out there | wer | 0.2381 |
| 320 | streaming_asr | EN_B00058_S06429_W000060 | 0.5461 | 98 | The alliance believed that parsh dis closure be very gigical to do with all of the data that was help there | wer | 0.2381 |
| 640 | streaming_asr | EN_B00058_S06429_W000060 | 0.5559 | 96 | The alliance believed that parsh dislosure be very gigical to do with all of the data that was out there | wer | 0.1905 |
| 1280 | streaming_asr | EN_B00058_S06429_W000060 | 0.5592 | 96 | The alliance believed that parsh dislosure be very gigical to do with all of the data that was helped there | wer | 0.2381 |
| 160 | streaming_asr | EN_B00058_S07483_W000027 | 0.8153 | 25 | And use the fix while high virulence problem | wer | 0.4444 |
| 320 | streaming_asr | EN_B00058_S07483_W000027 | 0.8089 | 25 | And use this to fix while high virulence problem | wer | 0.2222 |
| 640 | streaming_asr | EN_B00058_S07483_W000027 | 0.7452 | 27 | And use this to fix a high variance problem | wer | 0.0000 |
| 1280 | streaming_asr | EN_B00058_S07483_W000027 | 0.7197 | 31 | And use this to fix a high variance problem | wer | 0.0000 |
| 160 | streaming_asr | EN_B00058_S07511_W000000 | 0.4769 | 74 | Getting everybody out of the house and morning can be we really taught especially the first day school | wer | 0.2632 |
| 320 | streaming_asr | EN_B00058_S07511_W000000 | 0.4667 | 77 | Getting everybody out of the house and morning can be where really talk especially the first of school | wer | 0.2632 |
| 640 | streaming_asr | EN_B00058_S07511_W000000 | 0.4308 | 83 | Getting everybody out of the house and morning can be you really taught especially the first of school | wer | 0.2632 |
| 1280 | streaming_asr | EN_B00058_S07511_W000000 | 0.3897 | 91 | Getting everybody out of the house and morning can be very really taught especially the first time school | wer | 0.3158 |
| 160 | streaming_asr | EN_B00064_S08593_W000000 | 0.7270 | 105 | He teaches that the past does not exist A fact which belongs to the the sphere of knowledge and which therefore no one in the world can not | wer | 0.0741 |
| 320 | streaming_asr | EN_B00064_S08593_W000000 | 0.6980 | 110 | He teaches that the path does not exist A fact which belongs to the the sphere of knowledge and which therefore no one in the world can know | wer | 0.0741 |
| 640 | streaming_asr | EN_B00064_S08593_W000000 | 0.6911 | 114 | He teaches that the past does not exist a fact which belongs to the the sphere of knowledge and which therefore no one in the what world can know | wer | 0.0741 |
| 1280 | streaming_asr | EN_B00064_S08593_W000000 | 0.6928 | 116 | He teaches that the past does not exist a fact which belongs to the the sphere of knowledge and which therefore no one in the one world can know | wer | 0.0741 |
| 160 | streaming_asr | EN_B00083_S00689_W000013 | 0.5714 | 136 | You will see that using information entered the tool then calculates the net and as you use it budget all out for the apartment This of the number we need to keep below | wer | 0.3000 |
| 320 | streaming_asr | EN_B00083_S00689_W000013 | 0.5357 | 149 | You will say the you using information entered the tool then calculates the net energy used budget all around for apartment This of the number we need to keep below | wer | 0.3000 |
| 640 | streaming_asr | EN_B00083_S00689_W000013 | 0.5335 | 153 | You will see the you using information entered the tool then calculates the net energy usage budget allowed to for apartment This of the number we need to keep below | wer | 0.2333 |
| 1280 | streaming_asr | EN_B00083_S00689_W000013 | 0.5223 | 156 | You will see the you using information entered the tool then calculates the net energy usage budget allowed for apartment This of the number we need to keep below | wer | 0.2000 |
| 160 | streaming_asr | EN_B00083_S02942_W000001 | 0.6089 | 121 | It is going to be an resources I actually be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.2258 |
| 320 | streaming_asr | EN_B00083_S02942_W000001 | 0.5948 | 119 | It is going to be a resources as actually going to be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.1290 |
| 640 | streaming_asr | EN_B00083_S02942_W000001 | 0.6019 | 121 | It is going to be a resources as actually going to be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.1290 |
| 1280 | streaming_asr | EN_B00083_S02942_W000001 | 0.6089 | 120 | It is going to be a right resource as actually going to be a many course for any writers who want to learn how to write and publish their own serious stories | wer | 0.1290 |
| 160 | streaming_asr | EN_B00089_S01559_W000004 | 0.7934 | 37 | And every understood soaks or Lange are homework | wer | 0.6250 |
| 320 | streaming_asr | EN_B00089_S01559_W000004 | 0.7603 | 40 | And ever understood sox or Laungie or homework | wer | 0.5000 |
| 640 | streaming_asr | EN_B00089_S01559_W000004 | 0.7438 | 42 | And ever understood soaks or Lange are homework | wer | 0.6250 |
| 1280 | streaming_asr | EN_B00089_S01559_W000004 | 0.7562 | 42 | And ever understood soaks or Lange are homework | wer | 0.6250 |
| 160 | streaming_asr | EN_B00089_S03348_W000001 | 0.5220 | 119 | One of the things that like to do in each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.0690 |
| 320 | streaming_asr | EN_B00089_S03348_W000001 | 0.5161 | 118 | One of the things that out like to do when each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.1034 |
| 640 | streaming_asr | EN_B00089_S03348_W000001 | 0.5073 | 121 | One of the things that I like to do when each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.0690 |
| 1280 | streaming_asr | EN_B00089_S03348_W000001 | 0.5044 | 119 | One of the things that I like to do when each of my introduction astronomy classes is to be in the class with the astronomy picture of the day | wer | 0.0690 |
| 160 | streaming_asr | EN_B00091_S07092_W000002 | 0.5565 | 81 | And it comes off more like I kind uncle or a family member than Annie kind of authoritarian figured | wer | 0.2632 |
| 320 | streaming_asr | EN_B00091_S07092_W000002 | 0.5000 | 93 | But it comes off more like a current uncle or a family member than Annie kind of authoritarian figure | wer | 0.1053 |
| 640 | streaming_asr | EN_B00091_S07092_W000002 | 0.4879 | 92 | But it comes off more like a current uncle or a family member than Annie kind of authoritarian figure | wer | 0.1053 |
| 1280 | streaming_asr | EN_B00091_S07092_W000002 | 0.4758 | 97 | But it comes off more like a carrying uncle or a family member than Annie kind of authoritarian figure | wer | 0.1053 |
| 160 | streaming_asr | EN_B00091_S08343_W000001 | 0.7644 | 70 | But when the moment came to serenade my atreatment the gravations seremony | wer | 0.3333 |
| 320 | streaming_asr | EN_B00091_S08343_W000001 | 0.7382 | 71 | But when the moment came to serenade my at<\|glm_semantic_224\|>U<\|glm_semantic_4609\|>S there gravations seremony | wer | 0.4167 |
| 640 | streaming_asr | EN_B00091_S08343_W000001 | 0.7382 | 72 | But when the moment came to serenade my attempts the gravations seremony | wer | 0.3333 |
| 1280 | streaming_asr | EN_B00091_S08343_W000001 | 0.7225 | 72 | But when the moment came to serenade my attempts the gravations seremony | wer | 0.3333 |
| 160 | streaming_asr | EN_B00097_S02875_W000006 | 0.6259 | 151 | But the first thing is cross your legs because we don't want to want to receive from the lower part of our body We want to receive anything positive means we want to receive from the other upper part of our body | wer | 0.1026 |
| 320 | streaming_asr | EN_B00097_S02875_W000006 | 0.5957 | 160 | But uh for thing he's crossed your legs because we don't want to do receive from the law part of our body We want to receive anything positive means we want to receive from the upper part of our body | wer | 0.1282 |
| 640 | streaming_asr | EN_B00097_S02875_W000006 | 0.6082 | 151 | But uh first thing is cross your legs because we don't want to do receive from the lower part of our body We want to receive anything positive means we want to receive from the upper part of our body | wer | 0.0256 |
| 1280 | streaming_asr | EN_B00097_S02875_W000006 | 0.6082 | 151 | But uh first thing is cross your legs because we don't want to want receive from the lower part of our body We want to receive anything positive means we want to receive from the upper part of our body | wer | 0.0256 |
| 160 | streaming_asr | EN_B00097_S03849_W000000 | 0.5963 | 138 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 320 | streaming_asr | EN_B00097_S03849_W000000 | 0.5735 | 146 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 640 | streaming_asr | EN_B00097_S03849_W000000 | 0.5756 | 140 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 1280 | streaming_asr | EN_B00097_S03849_W000000 | 0.5549 | 141 | If you are not death don't and blind then you know that the American Boiswad democracy and capitalistic civilization other worst enemies of lab and progress | wer | 0.2222 |
| 160 | streaming_asr | HQ-Conversations_0000026067 | 0.7546 | 45 | 对还有很多内容服务的内容你就是在那种打线上的内容嗯 | cer | 0.5600 |
| 320 | streaming_asr | HQ-Conversations_0000026067 | 0.7500 | 48 | 对还有很多内容古佛的那种印度就是在那种大山羊的那种嗯 | cer | 0.2800 |
| 640 | streaming_asr | HQ-Conversations_0000026067 | 0.7639 | 46 | 对还有很多内容鼓舞的内容印度就是在那种大山羊的运动呢 | cer | 0.5200 |
| 1280 | streaming_asr | HQ-Conversations_0000026067 | 0.6944 | 60 | 但还有很多那种古法的那种建筑就是在那种搭山羊的那种嗯 | cer | 0.2000 |
| 160 | streaming_asr | HQ-Conversations_0000028308 | 0.7742 | 13 | 对兄弟咋了 | cer | 0.4000 |
| 320 | streaming_asr | HQ-Conversations_0000028308 | 0.7903 | 12 | 对兄弟咋了 | cer | 0.4000 |
| 640 | streaming_asr | HQ-Conversations_0000028308 | 0.7742 | 12 | 我想也咋办 | cer | 0.8000 |
| 1280 | streaming_asr | HQ-Conversations_0000028308 | 0.7258 | 16 | 我曾经也砸了 | cer | 1.2000 |
| 160 | streaming_asr | HQ-Conversations_0000041144 | 0.7942 | 52 | 对然后先伸出这个极端的其实我也有一点这种焦虑 | cer | 0.3043 |
| 320 | streaming_asr | HQ-Conversations_0000041144 | 0.7762 | 58 | 对然后下深处这个极端嘛其实我也有点这种焦虑 | cer | 0.3043 |
| 640 | streaming_asr | HQ-Conversations_0000041144 | 0.7762 | 56 | 对然后下深处这个极端嘛其实我也有一点这种焦虑 | cer | 0.2609 |
| 1280 | streaming_asr | HQ-Conversations_0000041144 | 0.7690 | 55 | 对然后下深处这个阶段嘛其实我也有一点这种焦虑 | cer | 0.1739 |
| 160 | streaming_asr | LibriSpeech_0000033920 | 0.5842 | 178 | And then the representatives who had been picked for everything but their grass of science and government when to penic over a myst of national prestige The space effort was turned over to the aircraft in industry | wer | 0.1944 |
| 320 | streaming_asr | LibriSpeech_0000033920 | 0.5627 | 179 | And when the representatives who had been picked for everything but their grass of science and government one in the panicked over a myst of national prestige The space effort was turned over to the aircraft in industry | wer | 0.1944 |
| 640 | streaming_asr | LibriSpeech_0000033920 | 0.5627 | 179 | And when the representatives who had been picked for everything but their grass of science and government one in the panicked over a myst of national prestige The space effort was turned over to the aircraft in industry | wer | 0.1944 |
| 1280 | streaming_asr | LibriSpeech_0000033920 | 0.5627 | 179 | And when the representatives who had been picked for everything but their grass of science and government one in the panicked over a miff of national prestige The space effort was turned over to the ear craft in industry | wer | 0.2500 |
| 160 | streaming_asr | LibriSpeech_0000035237 | 0.6345 | 139 | In that country there is a great deal of mass It covers the ground just as grass does here But the most interesting thing about these lemmings is the way they migrate | wer | 0.0312 |
| 320 | streaming_asr | LibriSpeech_0000035237 | 0.6176 | 143 | In that country where is a great deal of mass it covers the ground just as grass does here But the most interesting thing about these lemons is the way they migrate | wer | 0.0938 |
| 640 | streaming_asr | LibriSpeech_0000035237 | 0.6091 | 141 | In that country where is a great deal of mass it covers the ground just as grass does here But the most interesting thing about these lemons is the way they migrate | wer | 0.0938 |
| 1280 | streaming_asr | LibriSpeech_0000035237 | 0.6176 | 141 | In that country where is a great deal of mass it covers the ground just as grass does here But the most interesting thing about these lemons is the way they migrate | wer | 0.0938 |
| 160 | streaming_asr | LibriSpeech_0000068252 | 0.6833 | 165 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousands weppers have been collected | wer | 0.1471 |
| 320 | streaming_asr | LibriSpeech_0000068252 | 0.6725 | 170 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousands weppers have been collected | wer | 0.1471 |
| 640 | streaming_asr | LibriSpeech_0000068252 | 0.6523 | 175 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousands weppers have been collected | wer | 0.1471 |
| 1280 | streaming_asr | LibriSpeech_0000068252 | 0.6442 | 173 | The bureau of Health has transformed the city of Manila from a fifth infested hotbed of contagious diseases to one of the most healthful cities on the global six thousand lepers have been collected | wer | 0.0882 |
| 160 | streaming_asr | LibriSpeech_0000068284 | 0.7271 | 102 | Is not a world of facts but only a the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0714 |
| 320 | streaming_asr | LibriSpeech_0000068284 | 0.6919 | 107 | Is not a world of facts but only of the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0357 |
| 640 | streaming_asr | LibriSpeech_0000068284 | 0.7007 | 108 | Is not a world of facts but only of the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0357 |
| 1280 | streaming_asr | LibriSpeech_0000068284 | 0.7007 | 108 | Is not a world of facts but only of the meaning of facts It is a point of view for judging facts It appertains to a different alogy | wer | 0.0357 |
| 160 | streaming_asr | LibriSpeech_0000090398 | 0.6583 | 46 | How elegant how gentle she was and of what refined good manners the | wer | 0.0833 |
| 320 | streaming_asr | LibriSpeech_0000090398 | 0.6583 | 47 | How elegant how gentle she was and of what refined good manners the | wer | 0.0833 |
| 640 | streaming_asr | LibriSpeech_0000090398 | 0.6181 | 50 | How elegant how gentle she was and of what refined good manners | wer | 0.0000 |
| 1280 | streaming_asr | LibriSpeech_0000090398 | 0.6281 | 51 | How elegant how gentle she was and of what refined good manners | wer | 0.0000 |
| 160 | streaming_asr | LibriSpeech_0000090820 | 0.6442 | 186 | Which kept me in the how house fearnily too weeks The basement kitchen seemed heavily safe and warm in those days like a tight little boat in a winter sea The men were out in the fields all day hasking corn and when they came in noon | wer | 0.1458 |
| 320 | streaming_asr | LibriSpeech_0000090820 | 0.6132 | 195 | Which kept me in the how house for nearly two weeks The basement kitchen seemed heavenly safe and warm in those days like a tight little boat in a winter sea The men were out and the fields all day hasking corn and when they came in a noon | wer | 0.0833 |
| 640 | streaming_asr | LibriSpeech_0000090820 | 0.6199 | 194 | Which kept me in the how house for nearly two weeks The basement kitchen seemed heavily safe and warm in those days like a tight little boat in a winter sea The men were out and the fields all day husking corn and when they came in a noon | wer | 0.0833 |
| 1280 | streaming_asr | LibriSpeech_0000090820 | 0.6199 | 192 | Which kept me in the how house for nearly two weeks The basement kitchen seemed heavily safe and warm in those days like a tight little boat in a winter sea The men were out in the fields all day husking corn and when they came in a noon | wer | 0.0625 |
| 160 | streaming_asr | LibriSpeech_0000100601 | 0.6991 | 131 | In one enthusiastic jumble While Tom was off on his third rage my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 320 | streaming_asr | LibriSpeech_0000100601 | 0.7021 | 123 | In one enthusiastic jumble while Tom was off on his third rage my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 640 | streaming_asr | LibriSpeech_0000100601 | 0.6871 | 128 | In one enthusiastic jumble while Tom was off on his third rage my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 1280 | streaming_asr | LibriSpeech_0000100601 | 0.6841 | 133 | In one enthusiastic jumble while Tom was off on his third rade my attention was attracted by a man who stood a little apart looking as if his thoughts were far away | wer | 0.0312 |
| 160 | streaming_asr | LibriSpeech_0000104521 | 0.6789 | 184 | Until they drew up beside the palace steps And age to wink dressed in the uniform of silver cloth came forward to a sister them to allight said the scarecrow who his personage showed a set once to master the emperor | wer | 0.3250 |
| 320 | streaming_asr | LibriSpeech_0000104521 | 0.6718 | 183 | Until they drew up the beside the palace steps And aged to wink dressed in the uniform of silver cloth came forward to a sister them to allight Said the scarecrow who his personage showed a that once to your master the emperor | wer | 0.3000 |
| 640 | streaming_asr | LibriSpeech_0000104521 | 0.6754 | 182 | Until they drew up the beside the palace steps and aged to wink dressed in the uniform of silver cloth came forward to a sister them to allight said the scarecrow who his personage showed a that once to master the emperor | wer | 0.3250 |
| 1280 | streaming_asr | LibriSpeech_0000104521 | 0.6730 | 178 | Until they drew up the beside the palace steps and aged to wink dressed in the uniform of silver cloth came forward to assist them to allight said the scarecrow who his personage showed us at once to master the emperor | wer | 0.2250 |
| 160 | streaming_asr | LibriSpeech_0000124435 | 0.7206 | 152 | So now all the the children saw upon their plates apples sauce and squash and tomeadow and sweet potato and sour potato Not one of them could eat mouthful because not one was safed with meat | wer | 0.1892 |
| 320 | streaming_asr | LibriSpeech_0000124435 | 0.7018 | 154 | So now all the the children saw upon their plates apple sauce and squash and tomato and sweet potato and sour potato Not one of their and cody they mouthful because not one was safed with meat | wer | 0.2162 |
| 640 | streaming_asr | LibriSpeech_0000124435 | 0.6734 | 160 | So now although the children saw upon their plates apple sauce and squash and tomato and sweet potato and sour potato not one of their and cody they malful because not one was safed with the meat | wer | 0.1622 |
| 1280 | streaming_asr | LibriSpeech_0000124435 | 0.6721 | 163 | So now although the children saw upon their plates apple sauce and squash and tomato and sweet potato and sour potato not one of their and cody they malful because not one was satisfied with the meat | wer | 0.1351 |
| 160 | streaming_asr | LibriSpeech_0000124551 | 0.5954 | 200 | The platter family assembled in the cellar were a boat to begin breakfast where it was discovered that what of its members was missing Henry was the absent one I'd first through was but little notice taken of the circumstance | wer | 0.2051 |
| 320 | streaming_asr | LibriSpeech_0000124551 | 0.5855 | 203 | The platter family assembled in the cella were about to begin breakfast when it was discovered that one of its members was missing Henry was the absent one I'd first through was but little notice taken of the circumstance | wer | 0.1026 |
| 640 | streaming_asr | LibriSpeech_0000124551 | 0.5926 | 200 | The plants are family assembled in the cella were about to begin breakfast when it was discovered that one of its members was missing Henry was the absent one I'd first through was but little notice taken of the circumstance | wer | 0.1282 |
| 1280 | streaming_asr | LibriSpeech_0000124551 | 0.5783 | 202 | The planters family assembled in the cella were about to begin breakfast when it was discovered that one of its members was missing Henry was the absent one And first there was but little notice taken of the circumstance | wer | 0.0769 |
| 160 | streaming_asr | LibriSpeech_0000158589 | 0.6466 | 178 | I always did think Mary Harris ressembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Blackett for after receiving the affectionate greetings of near the whole company | wer | 0.1944 |
| 320 | streaming_asr | LibriSpeech_0000158589 | 0.6260 | 183 | I always did think Mary Harris ressembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Plathet for after receiving the affectionate greetings of nearly the whole company | wer | 0.1944 |
| 640 | streaming_asr | LibriSpeech_0000158589 | 0.6192 | 179 | I always did think Mary Harris ressembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Plattet for after receiving the affectionate greetings of near the whole company | wer | 0.2222 |
| 1280 | streaming_asr | LibriSpeech_0000158589 | 0.6315 | 177 | I always did think Mary Harris resembled a Chinese Mary Harris was pretty as a child I remember said the pleasant voice of Mrs Plattet for after receiving the affectionate greetings of nearly the whole company | wer | 0.1667 |
| 160 | streaming_asr | LibriSpeech_0000158773 | 0.5879 | 218 | All men in joy the blessings of the liberty of lived that by utilizing my capital at my my meek little income and I began to win money on security relying on my threat my judgment and my knowledge of the world I chose this business in preferance to all others | wer | 0.2449 |
| 320 | streaming_asr | LibriSpeech_0000158773 | 0.5597 | 221 | All men in joy the blessings of the liberty have lived that by utilizing my capital at my might make a little income in IBM began to win money on security relying on my threat my judgment and my knowledge of the world I choose this business in preferance to all others | wer | 0.2653 |
| 640 | streaming_asr | LibriSpeech_0000158773 | 0.5490 | 228 | All men in joy the blessings of the liberty of lived that by utilizing my capital at my my make little income and I'd be going to win money on security relying on my thrift my judgment and my knowledge of the world I chose this business in preferance to all others | wer | 0.2857 |
| 1280 | streaming_asr | LibriSpeech_0000158773 | 0.5477 | 228 | All men in joy the blessings of the liberty of lived that by utilizing my capital at my might make a little income and I'd be going to win money on security relying on my thrift my judgment and my knowledge of the world I chose this business in preferance to all others | wer | 0.2449 |
| 160 | streaming_asr | LibriSpeech_0000168081 | 0.6591 | 190 | Business getting arguments discernance by boards of directors considerations of corporate policy all of which influenced the political American and economic maps of the world I use really results of careful though in formal conversation | wer | 0.2353 |
| 320 | streaming_asr | LibriSpeech_0000168081 | 0.6187 | 209 | Business getting arguments decisions by boards of directors considerations of corporate policy all of which influence the political American and economic maps of the world I use the results of careful though in formal conversation | wer | 0.1471 |
| 640 | streaming_asr | LibriSpeech_0000168081 | 0.6149 | 209 | Business getting arguments decisions by boards of directors considerations of corporate policy all of which influence the political American and economic maps of the world a use real results of careful though in formal conversation | wer | 0.1765 |
| 1280 | streaming_asr | LibriSpeech_0000168081 | 0.5934 | 213 | Business getting arguments decisions by boards of directors considerations of corporate policy all of which influence the political American and economic maps of the world a usually the results of careful though in formal conversation | wer | 0.1176 |
| 160 | streaming_asr | LibriSpeech_0000215121 | 0.6580 | 166 | And return by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Want to crystall another to his head in talk of offensive But that's his not all continued Dumbla | wer | 0.3611 |
| 320 | streaming_asr | LibriSpeech_0000215121 | 0.6594 | 167 | And returned by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Wanted Christ another to his head in talking of offensive But that's his not all continued Dumbla | wer | 0.3056 |
| 640 | streaming_asr | LibriSpeech_0000215121 | 0.6449 | 172 | And returned by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Wanted Christ another to his head in Ockham and of offensive But that's his not all continued Dumbla | wer | 0.3333 |
| 1280 | streaming_asr | LibriSpeech_0000215121 | 0.6435 | 172 | And returned by you to me with your ind endorsement of course I immediately counted him over the forty bank notes Wants acrys to another to his head in talk of offensive But that's his not all continued Dumblaugh | wer | 0.3333 |
| 160 | streaming_asr | LibriSpeech_0000238719 | 0.6698 | 94 | So tray is the world to go on If this kind of thing be permitted I may be going out to dinner or to the opposite to night | wer | 0.1071 |
| 320 | streaming_asr | LibriSpeech_0000238719 | 0.6437 | 100 | Chao tray is the world to go on If this kind of thing be permitted I may be gilling out to dinner or to the oper to night | wer | 0.1429 |
| 640 | streaming_asr | LibriSpeech_0000238719 | 0.6271 | 100 | How tray is the world to go on if this kind of thing be permitted I may be going out to dinner or to the oper to night | wer | 0.0714 |
| 1280 | streaming_asr | LibriSpeech_0000238719 | 0.6247 | 99 | How tray is the world to go on if this kind of thing be permitted I may be going out to dinner or to the oper to night | wer | 0.0714 |
| 160 | streaming_asr | LibriSpeech_0000265196 | 0.6429 | 175 | I shall be really great if you will say nothing about this There are some in the house and neighborhood who are sitting enough as it is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.1111 |
| 320 | streaming_asr | LibriSpeech_0000265196 | 0.6268 | 174 | Are she a really great if you will say nothing about this There are some in the house and neighborhood who are sitting enough and is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.2222 |
| 640 | streaming_asr | LibriSpeech_0000265196 | 0.6224 | 171 | I show be really great if you will say nothing about this there are some in the house and neighborhood who are silly enough as it is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.1111 |
| 1280 | streaming_asr | LibriSpeech_0000265196 | 0.6210 | 174 | I shall be really great if you will say nothing about this There are some in the house and neighborhood who are silly enough as it is You stay here and if you do not feel inclined to go to bed read here a book | wer | 0.0889 |
| 160 | streaming_asr | LibriSpeech_0000271006 | 0.6465 | 133 | And not make an attempt to get money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 320 | streaming_asr | LibriSpeech_0000271006 | 0.6282 | 134 | And not make any attempt to get my money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 640 | streaming_asr | LibriSpeech_0000271006 | 0.6245 | 136 | And not make any attempt to get my money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 1280 | streaming_asr | LibriSpeech_0000271006 | 0.6447 | 131 | And not make any attempt to get my money for here is quite sure that I would never get more than enough to pay my traveling expenses I thank him for his advice | wer | 0.1562 |
| 160 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7709 | 34 | As still late now let's just get this over with | wer | 0.2000 |
| 320 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7621 | 33 | As too late now let's just get this over way | wer | 0.2000 |
| 640 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7577 | 34 | As too late now let's just get this over way | wer | 0.2000 |
| 1280 | streaming_asr | NCSSD_R_EN_0000000261 | 0.7797 | 33 | As jewell laid now let just get this over way | wer | 0.5000 |
| 160 | streaming_asr | VCTK_0000006134 | 0.6647 | 37 | There's influence to the ident is like an arrow | wer | 0.2222 |
| 320 | streaming_asr | VCTK_0000006134 | 0.6706 | 38 | This influence to the audience is like an arrow | wer | 0.0000 |
| 640 | streaming_asr | VCTK_0000006134 | 0.6941 | 36 | This influence to the audience is like an arrow | wer | 0.0000 |
| 1280 | streaming_asr | VCTK_0000006134 | 0.6647 | 38 | This influence to the audience is like an arrow | wer | 0.0000 |
| 160 | streaming_asr | VCTK_0000029186 | 0.8079 | 24 | It's going to be new challenge | wer | 0.1429 |
| 320 | streaming_asr | VCTK_0000029186 | 0.8146 | 21 | It's going to be new challenge | wer | 0.1429 |
| 640 | streaming_asr | VCTK_0000029186 | 0.8079 | 23 | It's going to be new challenge | wer | 0.1429 |
| 1280 | streaming_asr | VCTK_0000029186 | 0.8212 | 21 | It's going to be new challenge | wer | 0.1429 |
| 160 | streaming_asr | VCTK_0000029362 | 0.8500 | 18 | The body is exhausted | wer | 0.2500 |
| 320 | streaming_asr | VCTK_0000029362 | 0.8562 | 17 | The body is exhausted | wer | 0.2500 |
| 640 | streaming_asr | VCTK_0000029362 | 0.8500 | 19 | My body is exhausted | wer | 0.0000 |
| 1280 | streaming_asr | VCTK_0000029362 | 0.8625 | 20 | My body is exhausted | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0003918097 | 0.7019 | 126 | 哦是然后前两天呢就是就是为什么这个话题成为了一个大家争争香来说的一个热点呢就是这是其实你们应该 | cer | 0.0426 |
| 320 | streaming_asr | emilia_zh_0003918097 | 0.7061 | 127 | 哦是然后前两天呢这是就是为什么这个话题成为了一个大家争争三来说的一个热点呢就是这是其实你们应该 | cer | 0.0638 |
| 640 | streaming_asr | emilia_zh_0003918097 | 0.7167 | 122 | 哦是然后前两天呢就是就是为什么这个话题成为了一个大家争争香来说的一个热点呢就是这是其实你们应该 | cer | 0.0426 |
| 1280 | streaming_asr | emilia_zh_0003918097 | 0.7146 | 125 | 哦是然后前两天呢就是就是为什么这个话题成为了一个大家争争香来说的一个热点呢就是这是其实你们应该 | cer | 0.0426 |
| 160 | streaming_asr | emilia_zh_0003918326 | 0.7212 | 78 | 啊我我咱们还有一个点听一次这是这个事情连连起来就这些人他们演互相关联 | cer | 0.3611 |
| 320 | streaming_asr | emilia_zh_0003918326 | 0.6827 | 84 | 啊哦我想到还有一个点听一次就是这事情能连起来就这些人他们演互相关联 | cer | 0.2778 |
| 640 | streaming_asr | emilia_zh_0003918326 | 0.6699 | 85 | 啊我我想到还有一个点挺有意思的就是这些东西是人脸起来就这些人他们也互相关联 | cer | 0.0833 |
| 1280 | streaming_asr | emilia_zh_0003918326 | 0.6474 | 90 | 啊我我想到还有一个点你听有意思就是这些东西是人脸起来就这些人他们也互相关联 | cer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0004002573 | 0.5871 | 160 | 啊因为没有因为他这样有些人可能喜欢看你的那个那种然后他家的那份粉丝然后你换那种那种去拍他可能就不喜欢你这个那种他可能就会去取消 | cer | 0.2167 |
| 320 | streaming_asr | emilia_zh_0004002573 | 0.5752 | 165 | 啊因为没有因为他叫有些人可能喜欢看你那个那种然后他家的那种粉丝然后你换那种那种去拍他可能就不喜欢你这个那种他可能就会去取消 | cer | 0.2333 |
| 640 | streaming_asr | emilia_zh_0004002573 | 0.5585 | 165 | 啊因为没有因为他叫有限人可能喜欢看你的一个内容然后他加的一份词然后你换那种那种去拍他可能就不喜欢你这个内容他可能就会取消 | cer | 0.2167 |
| 1280 | streaming_asr | emilia_zh_0004002573 | 0.5465 | 171 | 啊因为因为因为他叫有限人可能喜欢看你的一个内容然后他家的就粉丝然后你换那种那种去拍他可能就不喜欢你这个内容他可能就会取消 | cer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0004003103 | 0.6905 | 211 | 所以我就觉得这些可能父母给我的影响会他会影响我但是我没有没有一套观念说我是必须得是反抗什么都去去争取我要的东西的我从来都是觉得说我可以很顺其自然自然而然是得到我想要的东西嗯 | cer | 0.0805 |
| 320 | streaming_asr | emilia_zh_0004003103 | 0.6697 | 225 | 所以我就觉得之前可能父母给我的影响会他会影响我但是我从来没有没有一套观念说我是必须得去反抗什么都去去争取我要的东西的我从来都是觉得说我可以很顺其自然自然而然是得到我想要的东西嗯 | cer | 0.0690 |
| 640 | streaming_asr | emilia_zh_0004003103 | 0.6580 | 228 | 所以我就觉得之前可能父母给我的影响会他会影响我但是我从来没有没有一套观念说我是必须得是反抗什么都去去争取我要的东西的我从来都是觉得说我可以很顺其自然自然而然是得到我想要的东西嗯 | cer | 0.0805 |
| 1280 | streaming_asr | emilia_zh_0004003103 | 0.6632 | 230 | 所以我就觉得这些可能父母给我的影响会他会影响我但是我从来没有没有一套观念说我是必须得去反抗什么都去去争取我要都信的我从来都是觉得说我可以很顺其自然自然而然是得到我想要懂行嗯 | cer | 0.1149 |
| 160 | streaming_asr | emilia_zh_0004036114 | 0.7151 | 51 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004036114 | 0.7097 | 49 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004036114 | 0.6989 | 54 | 没有上我的大学家里情况可以说是一言难尽 | cer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0004036114 | 0.7043 | 51 | 没有上过大学家里情况可以说是一言难尽 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004064583 | 0.8006 | 54 | 我相信每一次巧合都是一道资讯一个线索告诉我们 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004064583 | 0.7850 | 56 | 我相信每一次巧合都是一道资讯一个线索告诉我们 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004064583 | 0.7757 | 61 | 我相信每一次巧合都是一道自信一个线索告诉我们 | cer | 0.0909 |
| 1280 | streaming_asr | emilia_zh_0004064583 | 0.7850 | 58 | 我相信每一次巧合都是一道资讯一个线索告诉我们 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004064952 | 0.7282 | 50 | 也在国老下依然在我的脑子里甚至嘴上的关 | cer | 0.4211 |
| 320 | streaming_asr | emilia_zh_0004064952 | 0.7179 | 48 | 要在国若下依然在我的脑子里闪着贼亮的光 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0004064952 | 0.7179 | 52 | 压在国虏下依然在我的脑子里闪着贼亮的光 | cer | 0.1053 |
| 1280 | streaming_asr | emilia_zh_0004064952 | 0.7077 | 53 | 压在鼓篓下依然在我的脑子里闪着嘴亮的光 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0004111317 | 0.8029 | 143 | 我有个主意我们为什么不挖坑找水呢啊哦是的我相信如果我们挖的足够深我们可以找到水的东西那我们选择一个地点然后开始挖挖 | cer | 0.1071 |
| 320 | streaming_asr | emilia_zh_0004111317 | 0.7993 | 150 | 我有个主意我们为什么不挖坑找水呢哦是的我相信如果我们挖的足够深我们可以找到水的堆那我们选择一个地点然后开始挖挖吧 | cer | 0.0714 |
| 640 | streaming_asr | emilia_zh_0004111317 | 0.7896 | 151 | 我有个主意我们为什么不挖坑找水呢哦是的我相信如果我们挖的足够深我们可以找到水的堆认为我们选择一个地点然后开始挖挖吧 | cer | 0.0893 |
| 1280 | streaming_asr | emilia_zh_0004111317 | 0.7908 | 157 | 我有个主意我们为什么不挖坑找水呢哦是的我相信如果我们挖的足够深我们可以找到水的堆认为我们选择一个地点然后开始挖挖吧 | cer | 0.0893 |
| 160 | streaming_asr | emilia_zh_0004111570 | 0.7541 | 101 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004111570 | 0.7377 | 102 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004111570 | 0.7400 | 101 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004111570 | 0.7377 | 102 | 我的意思是没有哪个地方或者维度是我们被永远禁锢其中的要那样的地方来干什么呢 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004129851 | 0.7716 | 35 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004129851 | 0.7716 | 36 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004129851 | 0.7716 | 36 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004129851 | 0.7654 | 37 | 只见有两个人正坐在地上喝酒呢 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004130152 | 0.7834 | 57 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004130152 | 0.7762 | 56 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004130152 | 0.7870 | 55 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004130152 | 0.7798 | 56 | 现在所拥有的片刻的安乐瞬间将变成痛苦例如 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004176058 | 0.8522 | 32 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004176058 | 0.8478 | 35 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004176058 | 0.8391 | 35 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004176058 | 0.8478 | 33 | 新一代传奇发明改变了美国 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004176769 | 0.7401 | 40 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004176769 | 0.7232 | 44 | 最终我们会从床上爬起来找到点事情做 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0004176769 | 0.7401 | 41 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004176769 | 0.7288 | 44 | 最终我们会从床上爬起来找点事情做 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004212211 | 0.7339 | 75 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 320 | streaming_asr | emilia_zh_0004212211 | 0.7125 | 81 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 640 | streaming_asr | emilia_zh_0004212211 | 0.7217 | 79 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 1280 | streaming_asr | emilia_zh_0004212211 | 0.7156 | 78 | 这里表露的显而易见的挫败感感源于卡尔的个人经历而不是政治信念本身 | cer | 0.0645 |
| 160 | streaming_asr | emilia_zh_0004270141 | 0.7736 | 67 | 他的小山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0385 |
| 320 | streaming_asr | emilia_zh_0004270141 | 0.7673 | 68 | 他的笑山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0769 |
| 640 | streaming_asr | emilia_zh_0004270141 | 0.7547 | 71 | 他的小山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0385 |
| 1280 | streaming_asr | emilia_zh_0004270141 | 0.7484 | 73 | 他的小山上已经做了二十天了一直在注视着海面等着他回来 | cer | 0.0385 |
| 160 | streaming_asr | emilia_zh_0004270182 | 0.7827 | 67 | 知道的从前埋藏的水晶很久很久的骨头弹走工具螃蟹的老鼠们 | cer | 0.2963 |
| 320 | streaming_asr | emilia_zh_0004270182 | 0.7649 | 73 | 找到的从前埋藏的水晶很久很久的骨头弹走工具螃蟹的老鼠们 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004270182 | 0.7619 | 72 | 找到的从前埋藏的雨晶很久很久的骨头弹走工具螃蟹的老鼠们 | cer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004270182 | 0.7589 | 72 | 找到的从前埋藏的雨晶很久很久的骨头赶走工具螃蟹的老鼠们 | cer | 0.1852 |
| 160 | streaming_asr | emilia_zh_0004344705 | 0.8257 | 34 | 啊和mark正被一根粗粗的树枝缠绕着 | cer | 0.3529 |
| 320 | streaming_asr | emilia_zh_0004344705 | 0.7982 | 42 | 啊和马克正被一根粗粗的树枝缠绕着 | cer | 0.1176 |
| 640 | streaming_asr | emilia_zh_0004344705 | 0.7844 | 43 | 安娜和马克正被一根粗粗的树枝缠绕着 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004344705 | 0.7661 | 47 | 安娜和马克正被一根粗粗的树枝缠绕着 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004358307 | 0.6534 | 124 | You was up to him to prove himself for there's six two make them proud of him and his music without the fate idea of how proud they were already | wer | 0.1333 |
| 320 | streaming_asr | emilia_zh_0004358307 | 0.6375 | 129 | It was a to him to prove himself for there's six two make them proud of him and his music without the fateless idea of how proud they were already | wer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0004358307 | 0.6315 | 127 | It was up to him to prove himself for their sakes to make them proud of him and his music without the fateless idea of how proud they were already | wer | 0.0667 |
| 1280 | streaming_asr | emilia_zh_0004358307 | 0.6275 | 128 | It was up to him to prove himself for their sakes to make them proud of him and his music without the fateless idea of how proud they were already | wer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0004358957 | 0.7912 | 82 | 我定的样子足够看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.2069 |
| 320 | streaming_asr | emilia_zh_0004358957 | 0.7690 | 85 | 我听这鸭子足足看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.1379 |
| 640 | streaming_asr | emilia_zh_0004358957 | 0.7715 | 85 | 我听这鸭子足足看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.1379 |
| 1280 | streaming_asr | emilia_zh_0004358957 | 0.7666 | 84 | 我盯着鸭子足足看了几秒钟之后这才嘴巴一张开口问道女子怎么知道 | cer | 0.0690 |
| 160 | streaming_asr | emilia_zh_0004422761 | 0.6809 | 44 | This led many twire people to settle into towns and cities | wer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0004422761 | 0.6489 | 45 | This led many twara people to settle into towns and cities | wer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0004422761 | 0.6543 | 44 | This led many twara people to settle into towns and cities | wer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0004422761 | 0.6543 | 44 | This led many twara people to settle in towns and cities | wer | 0.0909 |
| 160 | streaming_asr | emilia_zh_0004472880 | 0.7088 | 50 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 320 | streaming_asr | emilia_zh_0004472880 | 0.7198 | 44 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 640 | streaming_asr | emilia_zh_0004472880 | 0.7088 | 48 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 1280 | streaming_asr | emilia_zh_0004472880 | 0.7198 | 46 | 那可不见得就是为了热的时候打这山两块 | cer | 0.2632 |
| 160 | streaming_asr | emilia_zh_0004519522 | 0.7640 | 33 | 这这个姓王的客人呢呃是昨天来 | cer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0004519522 | 0.7640 | 35 | 对这位姓王的客人呢呃是昨天来 | cer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0004519522 | 0.7640 | 33 | 对这位姓王的客人呢呃是昨天来 | cer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0004519522 | 0.7764 | 31 | 这这位姓王的客人呢呃是昨天来的 | cer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0004519646 | 0.7742 | 47 | 伸出毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004519646 | 0.7604 | 47 | 伸出毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004519646 | 0.7558 | 46 | 伸出毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004519646 | 0.7604 | 47 | 身处毛茸茸的小爪子摸了摸盒子里的东西 | cer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0004621436 | 0.6901 | 45 | But mother wrote and asked me if they could possibly come as paying guess | wer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0004621436 | 0.6244 | 51 | They mother wrote and asked me if they could possibly come as paying guess | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004621436 | 0.6291 | 50 | They mother wrote and asked me they could possibly come as paying guests | wer | 0.1429 |
| 1280 | streaming_asr | emilia_zh_0004621436 | 0.6103 | 56 | There mother wrote and asked me if they could possibly come as paying guests | wer | 0.0714 |
| 160 | streaming_asr | emilia_zh_0004633795 | 0.5940 | 88 | We have a lively game of prosy one to corner on the flat left top with the little tree for basis | wer | 0.3500 |
| 320 | streaming_asr | emilia_zh_0004633795 | 0.5436 | 86 | We have a lively game of prosy one to corner on the flat left top with the little tree for basis | wer | 0.3500 |
| 640 | streaming_asr | emilia_zh_0004633795 | 0.5436 | 86 | We had a lively game of prosy one to corner on the flat left top with the little tree for basis | wer | 0.3000 |
| 1280 | streaming_asr | emilia_zh_0004633795 | 0.5537 | 87 | We had a lively game of prosy ones of corner on the flat plustop with the little tree for basis | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004659190 | 0.6438 | 93 | Seeing the the wind was close to granting what he replaced When you are loved you can you anything creation | wer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0004659190 | 0.6306 | 95 | Seeing the the wind was close to granting what he replaced When you are loved you can do anything creation | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004659190 | 0.6095 | 99 | Seeing the the wind was close to granting what he replaced When you are loved you can do anything in creation | wer | 0.0952 |
| 1280 | streaming_asr | emilia_zh_0004659190 | 0.6121 | 102 | Seeing that the wind was close to granting what he replaced When you are loved you can you anything in creation | wer | 0.0952 |
| 160 | streaming_asr | emilia_zh_0004692799 | 0.5991 | 188 | So we taken these strings instrument instead of using the them as these warm of broad filled communicators we're in the string family We're sort of turing them in now two procussion instrument That's a very procussive effect | wer | 0.3243 |
| 320 | streaming_asr | emilia_zh_0004692799 | 0.5697 | 190 | So we taken these string instrument instead of using the them as these warm of broad filled communicators were in the string family where there's a lot turning them in now to protection instrument That's a very procussive effect | wer | 0.3243 |
| 640 | streaming_asr | emilia_zh_0004692799 | 0.5650 | 194 | So we taken these string instrument instead of using the them as these warm of brato filled communicators were in the string family where there's a lot treme them in now to protection instrument That's a very procussive effect | wer | 0.3514 |
| 1280 | streaming_asr | emilia_zh_0004692799 | 0.5573 | 196 | So we taken these string instrument instead of using the them as these warm of broad filled communicators were in the string family where there's a tongue them in now to protection instrument That's a very procussive effect | wer | 0.3243 |
| 160 | streaming_asr | emilia_zh_0004706082 | 0.6708 | 35 | The quality of looking clever that he was | wer | 0.2222 |
| 320 | streaming_asr | emilia_zh_0004706082 | 0.6211 | 38 | The quality of looking clever then he was | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004706082 | 0.6087 | 40 | The quality of looking clever then he was | wer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004706082 | 0.6398 | 38 | The quality of looking clever then he was | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004724705 | 0.5731 | 103 | And while I had to run it when we started I had d submitted that we really talking about what seems like wanting physical to first | wer | 0.3750 |
| 320 | streaming_asr | emilia_zh_0004724705 | 0.5234 | 113 | And while I had to open one we simulated I had some that were really talking about what seems like wanting physical to first | wer | 0.3750 |
| 640 | streaming_asr | emilia_zh_0004724705 | 0.5175 | 112 | And while I had to manage when we surrendered I had to submit that we're really talking about what seems like wanting physical to first | wer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004724705 | 0.5175 | 111 | And while I had to open the end research I had submit that we're really talking about what seems like wanting physical to first | wer | 0.3750 |
| 160 | streaming_asr | emilia_zh_0004724727 | 0.7354 | 91 | Is it traditional in out But as you should for today the new things is uh speak with a a other for new | wer | 0.3913 |
| 320 | streaming_asr | emilia_zh_0004724727 | 0.7375 | 86 | Is it traditional in out But a thing so for today the new things is a speaking with a a other for new | wer | 0.3478 |
| 640 | streaming_asr | emilia_zh_0004724727 | 0.7289 | 85 | Is it traditional in our but uh I still for today the new things is uh speak with a a other for new | wer | 0.3913 |
| 1280 | streaming_asr | emilia_zh_0004724727 | 0.7462 | 81 | Is it traditional in our but uh the military for today the new things is uh speak with a a other type new | wer | 0.3913 |
| 160 | streaming_asr | emilia_zh_0004732647 | 0.5436 | 99 | He wanted to find out what made the noiseers that he people were afraid of and there were no nothing in the caves to tell him | wer | 0.2083 |
| 320 | streaming_asr | emilia_zh_0004732647 | 0.5369 | 101 | He wanted to find out what made the noiseers that he people were afraid of and there were no nothing in the caves to tell him | wer | 0.2083 |
| 640 | streaming_asr | emilia_zh_0004732647 | 0.5336 | 96 | He wanted to find out what made the noiseers that he people were afraid of and there were no nothing in the caves to tell him | wer | 0.2083 |
| 1280 | streaming_asr | emilia_zh_0004732647 | 0.5302 | 98 | He wanted to find out what made the noiseers that he people were afraid of and there were was nothing in the caves to tell him | wer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0004754634 | 0.6667 | 40 | Perhaps there is second joke suggested the jack door | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0004754634 | 0.5882 | 44 | Perhaps may have a second joke suggested the jack or | wer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0004754634 | 0.6144 | 43 | Perhaps there are a second joke suggested the jack door | wer | 0.1000 |
| 1280 | streaming_asr | emilia_zh_0004754634 | 0.5948 | 46 | Perhaps there is second joke suggested the jack or | wer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0004754844 | 0.6358 | 40 | Five stuff these clouds here and then came to the earth | wer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0004754844 | 0.6093 | 42 | Five stuff these clives here and then came to the earth | wer | 0.2727 |
| 640 | streaming_asr | emilia_zh_0004754844 | 0.6291 | 40 | Five stuff these clouds here and then came to the earth | wer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0004754844 | 0.5960 | 43 | Found stuff these clouds here and then came to the earth | wer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0004776929 | 0.7439 | 59 | In the evening they stefled down for a big family dinner This was eat | wer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0004776929 | 0.6877 | 65 | In evening they stessled down for a big family dinner This was it | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004776929 | 0.7158 | 56 | In the evening they settled down for a big family dinner This was it | wer | 0.0714 |
| 1280 | streaming_asr | emilia_zh_0004776929 | 0.7123 | 58 | In the evening they st settled down for a big family dinner This was it | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0004777075 | 0.6737 | 74 | He appeared over Dink's shoulder at the someday newspaper So what's the next day Kami Tuesday | wer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0004777075 | 0.6556 | 78 | He appeared over dink shoulder at the someday newspaper So what's the next day Kami Tuesday | wer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0004777075 | 0.6465 | 79 | He perished over dink shoulder at the someday newspaper So what's the next day Kami Tuesday | wer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0004777075 | 0.6677 | 76 | He perished over dink shoulder at the someday newspaper so what's the next day Kami Tuesday | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004797638 | 0.8579 | 25 | Why can't kill fire Thank you not quickly Try | wer | 0.5556 |
| 320 | streaming_asr | emilia_zh_0004797638 | 0.8421 | 25 | Why thank you for thank you not quickly Try | wer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0004797638 | 0.8263 | 27 | Fine thank you fine Thank you not quickly Try | wer | 0.1111 |
| 1280 | streaming_asr | emilia_zh_0004797638 | 0.8053 | 25 | Fine thank you thank thank you not quickly try | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004797649 | 0.8377 | 23 | In no Tony has got the word catch | wer | 0.2500 |
| 320 | streaming_asr | emilia_zh_0004797649 | 0.8115 | 26 | In now Tony has got the word catch | wer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0004797649 | 0.8115 | 25 | In no tony has got the word catch | wer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0004797649 | 0.8063 | 26 | In a tony has got the word catch | wer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0004804632 | 0.5920 | 60 | It is often a pretty sight when several of these boats are more together | wer | 0.0714 |
| 320 | streaming_asr | emilia_zh_0004804632 | 0.5287 | 61 | It is often a pretty sight when several of these both are more together | wer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0004804632 | 0.5115 | 62 | It is often a pretty sight when several of these both are more together | wer | 0.1429 |
| 1280 | streaming_asr | emilia_zh_0004804632 | 0.4770 | 65 | It is often a pretty sight when several of these both are more together | wer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0004804709 | 0.6490 | 41 | render the unsafe for the them to venture outside the town | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0004804709 | 0.6358 | 41 | render the unsafe for them to venture outside the town | wer | 0.1000 |
| 640 | streaming_asr | emilia_zh_0004804709 | 0.5894 | 47 | render the unsafe for them to venture outside the town | wer | 0.1000 |
| 1280 | streaming_asr | emilia_zh_0004804709 | 0.6026 | 46 | render it unsaved for them to venture outside the town | wer | 0.1000 |
| 160 | streaming_asr | emilia_zh_0004841011 | 0.6512 | 55 | I all those users who churning and resuracting have the lowfriend camps | wer | 0.6154 |
| 320 | streaming_asr | emilia_zh_0004841011 | 0.5907 | 57 | And all those users who are churning and resuracting have the lowfriend counts | wer | 0.3846 |
| 640 | streaming_asr | emilia_zh_0004841011 | 0.5907 | 56 | And all those users who are churning and resuracting have been loafromed camps | wer | 0.4615 |
| 1280 | streaming_asr | emilia_zh_0004841011 | 0.5674 | 61 | And all those users who are churning and resuracting have been loafed from camps | wer | 0.5385 |
| 160 | streaming_asr | emilia_zh_0004843191 | 0.6110 | 209 | Hence forth not only European drug refers but European scholars almost all over else of knowledge began to draw maps with spaces left to hello and they began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.1702 |
| 320 | streaming_asr | emilia_zh_0004843191 | 0.5718 | 221 | Hence forth not only European Geographers but European scholars almost all other else of knowledge began to draw maps with spaces left to hello and they began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.1064 |
| 640 | streaming_asr | emilia_zh_0004843191 | 0.5718 | 222 | Hence forth not only European Geographers but European scholars almost all other else of knowledge began to draw maps with spaces left to hello in They began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.0851 |
| 1280 | streaming_asr | emilia_zh_0004843191 | 0.5483 | 228 | Hence forth not only European Geographers but European scholars almost all other else of knowledge began to draw maps with spaces left to hello in They began to admit that their theories were not perfect and that they were important things that they did not know | wer | 0.0851 |
| 160 | streaming_asr | emilia_zh_0004843272 | 0.5907 | 151 | It's principle tenderness that I con dominant growth is this supreme good or it least approxy for the supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.3226 |
| 320 | streaming_asr | emilia_zh_0004843272 | 0.5722 | 156 | It's principle tennett is that I economic growth is the supreme good or it least approxy for the Supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.1935 |
| 640 | streaming_asr | emilia_zh_0004843272 | 0.5630 | 158 | It's principle tendency that I economic growth is this supreme good or it least approxy for the Supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0004843272 | 0.5704 | 154 | It's principle tennant is that I economic growth is this supreme good or it least approxy for the Supreme good because justice freedom and even happiness all depend on economic growth | wer | 0.2258 |
| 160 | streaming_asr | emilia_zh_0004873848 | 0.6979 | 40 | Margaret says little art with an honest open smile | wer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0004873848 | 0.7135 | 37 | Margaret says little up with an honest open smile | wer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0004873848 | 0.6927 | 37 | Margaret faced little up with an honest open smile | wer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0004873848 | 0.7083 | 37 | Margaret says little up with an honest open smile | wer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0004874290 | 0.6534 | 61 | The book from Mr Thornton arrived that evening with a kind note in Faid | wer | 0.3846 |
| 320 | streaming_asr | emilia_zh_0004874290 | 0.6096 | 64 | The book from Mr Thornton arrived that evening with a kind note in Faid | wer | 0.3846 |
| 640 | streaming_asr | emilia_zh_0004874290 | 0.6135 | 63 | The book from Mr Thornton arrived that evening with a kind note in Fied | wer | 0.3846 |
| 1280 | streaming_asr | emilia_zh_0004874290 | 0.6056 | 64 | The book from Mr Thornton arrived that evening with a kind note in Fied | wer | 0.3846 |
| 160 | streaming_asr | emilia_zh_0004879738 | 0.6038 | 44 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0004879738 | 0.5849 | 44 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 640 | streaming_asr | emilia_zh_0004879738 | 0.6038 | 41 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 1280 | streaming_asr | emilia_zh_0004879738 | 0.6038 | 41 | The resignations infuriated Elizabeth and Sunny | wer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0004880227 | 0.6783 | 64 | I shall always remember the hours I spent with the master of the house of Russia | wer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0004880227 | 0.6744 | 65 | I shall always remember the hours I spent with the master of the house of Russia | wer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0004880227 | 0.6357 | 64 | I shall always remember the hours I spent with the master of the house of Russia | wer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0004880227 | 0.6240 | 66 | I shall always remember the hours I spent with the master of the house of Usher | wer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0004927443 | 0.6878 | 59 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004927443 | 0.6787 | 62 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004927443 | 0.6968 | 57 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004927443 | 0.6878 | 58 | 但是美国人之所以不懂悠闲还有一个更重要的原因 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0004927721 | 0.7669 | 51 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 320 | streaming_asr | emilia_zh_0004927721 | 0.7585 | 52 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 640 | streaming_asr | emilia_zh_0004927721 | 0.7542 | 54 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0004927721 | 0.7585 | 52 | 一个人只有在臣服于这国能量时才能了解他 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0004943305 | 0.8385 | 50 | 我们国家层面面临历来都如此勉马十八周岁 | cer | 0.3810 |
| 320 | streaming_asr | emilia_zh_0004943305 | 0.8230 | 54 | 我们国家层面面临历来都如此勉马十八周岁啊 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0004943305 | 0.8230 | 55 | 我们国家层面面临历来都如此勉马十八周岁啊 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0004943305 | 0.8292 | 50 | 我们国家层面面临历来都如此勉蛮十八周岁啊 | cer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0004943713 | 0.7874 | 113 | 等因我知道他清楚了你在这上面删除了夕阳和乘客尝试问从业者黑洞狐狐他们那个情感的愚蠢 | cer | 0.4651 |
| 320 | streaming_asr | emilia_zh_0004943713 | 0.7721 | 118 | 当时因我知道他清楚了你在这上面三处的信仰和乘客的尝试问从业者奋斗无与乎他那个情感的愚蠢 | cer | 0.3488 |
| 640 | streaming_asr | emilia_zh_0004943713 | 0.7755 | 116 | 但是因我知道他清楚了你在这上面删除的信仰和乘客的尝试问从业者奋斗维护他们你的情感的愚蠢 | cer | 0.2093 |
| 1280 | streaming_asr | emilia_zh_0004943713 | 0.7806 | 114 | 但是因我知道他清楚了你在这上面删除的信仰和乘客尝试问从业者黑洞维护他们你的情感的愚蠢 | cer | 0.2791 |
| 160 | streaming_asr | emilia_zh_0004999877 | 0.7904 | 78 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0004999877 | 0.7854 | 78 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0004999877 | 0.7854 | 79 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0004999877 | 0.7904 | 78 | 我们就应该尽量依照佛所讲的方法去实施这样才会有进步和收效 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005000497 | 0.7679 | 58 | 鉴于企业所有的债务之后所得的就是企业价值 | cer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0005000497 | 0.7571 | 58 | 减去企业所有的债务之后所得的就是企业的价值 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005000497 | 0.7714 | 58 | 减去企业所有的债务之后所得的就是企业的价值 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005000497 | 0.7750 | 55 | 减去企业所有的债务之后所得的就是企业的价值 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005059483 | 0.7246 | 97 | 不是的目的如果仅仅说为了获取更大的答案不是也就不成为故事在这对于普通语言 | cer | 0.3714 |
| 320 | streaming_asr | emilia_zh_0005059483 | 0.7246 | 99 | 不是的目的如果仅仅是为了获取更大的答案不是也就不成为布什再者对普通员 | cer | 0.2571 |
| 640 | streaming_asr | emilia_zh_0005059483 | 0.7193 | 97 | 故事的目的如果仅仅是为了获取更大的答案不是也就不成为故事再者对于普通人 | cer | 0.2286 |
| 1280 | streaming_asr | emilia_zh_0005059483 | 0.7193 | 96 | 不是的目的如果仅仅是为了获取更大的答案不是也就不成为故事再者对于普通人 | cer | 0.2286 |
| 160 | streaming_asr | emilia_zh_0005094293 | 0.7204 | 55 | 对于托过程来说已经是一件很难做到的事情了 | cer | 0.1429 |
| 320 | streaming_asr | emilia_zh_0005094293 | 0.7109 | 57 | 但是错工程学来说已经是一件很难做到事情了 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005094293 | 0.7014 | 58 | 但是错误工程学来说已经是一件很难做到的事情了 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0005094293 | 0.6825 | 58 | 但是从工程学来说已经是一件很难做到的事情了 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0005094550 | 0.7323 | 107 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005094550 | 0.7279 | 113 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005094550 | 0.7212 | 116 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005094550 | 0.7146 | 116 | 他也需要运用充满矛盾含混不清的概念因此这个问题不花费较长的篇幅就不能够希望解释清楚 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005181378 | 0.6932 | 52 | 不是客气你记得我的几本书请你带回去 | cer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0005181378 | 0.6705 | 53 | 不是客气你记得我的几本书其实你带了回去 | cer | 0.2222 |
| 640 | streaming_asr | emilia_zh_0005181378 | 0.6761 | 52 | 不是客气你记得我的几本书其实你带了回去 | cer | 0.2222 |
| 1280 | streaming_asr | emilia_zh_0005181378 | 0.6591 | 54 | 不是客气你记得我的几本书其实你带了回去 | cer | 0.2222 |
| 160 | streaming_asr | emilia_zh_0005244297 | 0.6857 | 63 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005244297 | 0.6714 | 66 | 总得做点不一样的事情吧打电话报警还报出了<\|write_generate\|><\|cmn\|><\|start_content\|>车牌号 | cer | 1.8261 |
| 640 | streaming_asr | emilia_zh_0005244297 | 0.6714 | 67 | 总得做点不一样的事情吧打电话报警还爆出了车牌号 | cer | 0.0435 |
| 1280 | streaming_asr | emilia_zh_0005244297 | 0.6762 | 66 | 总得做点不一样的事情吧打电话报警还报出了车牌号 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005313494 | 0.6582 | 46 | 那不能读了吧因为后面没有一个应该的拼 | cer | 0.3000 |
| 320 | streaming_asr | emilia_zh_0005313494 | 0.6772 | 47 | 那不能读了吧因为后面没有一个应该的拼 | cer | 0.3000 |
| 640 | streaming_asr | emilia_zh_0005313494 | 0.6076 | 54 | 他不能读了吧因为后面没有一个应该的拼 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0005313494 | 0.5823 | 56 | 他就不能读了吧因为后面没有一个应该的拼 | cer | 0.2000 |
| 160 | streaming_asr | emilia_zh_0005347772 | 0.6654 | 85 | 另外民主党被剥夺了了死胡同据民主党来说他们面临这个困难的选择 | cer | 0.2333 |
| 320 | streaming_asr | emilia_zh_0005347772 | 0.6502 | 86 | 另外民主党被一波走进了死胡同对于民主党来说他们面临这个困难的选择 | cer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0005347772 | 0.6578 | 82 | 另外民主党被一波走进了死胡同对于民主党来说他们面临这个困难的选择 | cer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0005347772 | 0.6426 | 87 | 另外民主党被一波走进了死胡同对于民主党来说他们面临着困难的选择 | cer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0005370483 | 0.7803 | 90 | 独立有些呢就就就就是是是是是一个什么感觉呢就是鱼龙混搭但是这村好像是不是个好 | cer | 0.2683 |
| 320 | streaming_asr | emilia_zh_0005370483 | 0.7551 | 97 | 独立游戏呢就就就就是是是是一个什么感觉呢就是鱼龙混的当然这村好像的不是个什么 | cer | 0.2439 |
| 640 | streaming_asr | emilia_zh_0005370483 | 0.7712 | 93 | 独立游戏呢就就就就是是是是一个什么感觉呢就是鱼龙混的当然这村好像的不是个什么 | cer | 0.2439 |
| 1280 | streaming_asr | emilia_zh_0005370483 | 0.7574 | 97 | 独立游戏呢就就就就是是是是一个什么感觉呢就是鱼龙混搭当然这村好像的不是个什么 | cer | 0.2439 |
| 160 | streaming_asr | emilia_zh_0005370632 | 0.8085 | 37 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0005370632 | 0.7872 | 44 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0005370632 | 0.7787 | 44 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005370632 | 0.7787 | 42 | 你是看不到说你这个形态展发展下去 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0005420605 | 0.7602 | 41 | 有的高考是考一这只大一变如果太行的 | cer | 0.5294 |
| 320 | streaming_asr | emilia_zh_0005420605 | 0.7398 | 42 | 有的光卡是靠一这只大一变是太行的 | cer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0005420605 | 0.7398 | 43 | 有的光卡是靠一这只大一变是不太行的 | cer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0005420605 | 0.7398 | 43 | 有的高考是靠一这只大一变是不太行的 | cer | 0.3529 |
| 160 | streaming_asr | emilia_zh_0005421693 | 0.7273 | 39 | 这物件东西是它昨晚制作的交通工具 | cer | 0.1250 |
| 320 | streaming_asr | emilia_zh_0005421693 | 0.7208 | 41 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005421693 | 0.7143 | 43 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005421693 | 0.7078 | 43 | 这五件东西是他昨晚制作的交通工具 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005507035 | 0.7273 | 50 | 那么日本政府还会收回这个决定了大的很多关心 | cer | 0.1905 |
| 320 | streaming_asr | emilia_zh_0005507035 | 0.7059 | 54 | 那么日本政府还会收回这个决定了大的很多关心 | cer | 0.1905 |
| 640 | streaming_asr | emilia_zh_0005507035 | 0.6898 | 54 | 那么日本政府还会收回这个决定了大的很多关心 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0005507035 | 0.6845 | 55 | 那么日本政府还会收回这个决定了大的很关心 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0005507553 | 0.6524 | 41 | Will you walk with me in the water garden He said softly | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0005507553 | 0.6203 | 44 | Will you walk with me in the water garden He said softly | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0005507553 | 0.5829 | 49 | Will you walk with me in the water garden he said softly | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0005507553 | 0.5829 | 49 | Will you walk with me in the water garden he said softly | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0005578304 | 0.9205 | 26 | 对树叶归来就好奇的问 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0005578304 | 0.9178 | 26 | 对树叶归来就好奇的问 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005578304 | 0.9233 | 25 | 对树叶回来就好奇的问 | cer | 0.2500 |
| 1280 | streaming_asr | emilia_zh_0005578304 | 0.9233 | 23 | 你对树叶回来就好奇的问 | cer | 0.2500 |
| 160 | streaming_asr | emilia_zh_0005578734 | 0.7351 | 46 | 做给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 320 | streaming_asr | emilia_zh_0005578734 | 0.7297 | 46 | 都给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 640 | streaming_asr | emilia_zh_0005578734 | 0.7297 | 45 | 做给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0005578734 | 0.7189 | 46 | 做给我一些你你能给陪伴给陪伴能给爱给爱 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0005601476 | 0.7818 | 31 | 但情绪中不如把心情找过场 | cer | 0.3333 |
| 320 | streaming_asr | emilia_zh_0005601476 | 0.7818 | 32 | 把情绪中不如把心情找过场 | cer | 0.3333 |
| 640 | streaming_asr | emilia_zh_0005601476 | 0.7879 | 30 | 外情绪中不如把心情找个不好 | cer | 0.3333 |
| 1280 | streaming_asr | emilia_zh_0005601476 | 0.7879 | 30 | 外情绪中不如把心情找过场 | cer | 0.3333 |
| 160 | streaming_asr | emilia_zh_0005669352 | 0.7762 | 103 | 这天杀的一杀两口红狼头都抖动了一下帮助儿子落了一地哥哥爬上的东边的栗子一看是骗子货 | cer | 0.2683 |
| 320 | streaming_asr | emilia_zh_0005669352 | 0.7581 | 111 | 是天杀的一生两朵红狼头都抖动了一下绑着子儿落了一地哥哥爬上的东边的栗子一看是骗子货 | cer | 0.1951 |
| 640 | streaming_asr | emilia_zh_0005669352 | 0.7621 | 109 | 是听沙子一声两朵红狼头都抖动了一下绑着子儿落了一地哥哥爬上的东边的栗子一看是骗子货 | cer | 0.1463 |
| 1280 | streaming_asr | emilia_zh_0005669352 | 0.7601 | 106 | 是听沙子一声两朵红狼头都抖动了一下棒着子儿落了一地哥哥爬上了东边的栗子一看是骗子货 | cer | 0.0976 |
| 160 | streaming_asr | emilia_zh_0005670671 | 0.7921 | 75 | 时至今日且在人类社会的各类主流中所败的角色已经无可代替企业的生 | cer | 0.2424 |
| 320 | streaming_asr | emilia_zh_0005670671 | 0.7847 | 78 | 时至今日企业在人类社会的各类组织中所办的角色已经无可代替企业的生 | cer | 0.1212 |
| 640 | streaming_asr | emilia_zh_0005670671 | 0.7847 | 77 | 时至今日企业在人类社会的各类组织中所办的角色已经无可代替企业的生 | cer | 0.1212 |
| 1280 | streaming_asr | emilia_zh_0005670671 | 0.7871 | 75 | 时至今日企业在人类社会的各类组织中所办的角色意见无可代替企业的生 | cer | 0.0909 |
| 160 | streaming_asr | emilia_zh_0005748506 | 0.7388 | 135 | 这是我后来我觉得我理解这个意思他其实告诉你你你不要想那么多就是你有这一个呃想法的时候你你去做就好了你先去体验 | cer | 0.1176 |
| 320 | streaming_asr | emilia_zh_0005748506 | 0.7388 | 133 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这个呃想法的时候你你去做就好了你先去去 | cer | 0.0784 |
| 640 | streaming_asr | emilia_zh_0005748506 | 0.7285 | 137 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这个呃想法的时候你你去做就好了你先去去 | cer | 0.0784 |
| 1280 | streaming_asr | emilia_zh_0005748506 | 0.7251 | 138 | 这是我后来我觉得我理解这个意思他其实告诉你你不要想那么多就是你有这个呃想法的时候你你去做就好了你先去去 | cer | 0.0784 |
| 160 | streaming_asr | emilia_zh_0005749601 | 0.7390 | 94 | 是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了花第一点就是那功能不足 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0005749601 | 0.7209 | 97 | 就是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了的话第一点就是认知功能不足 | cer | 0.1591 |
| 640 | streaming_asr | emilia_zh_0005749601 | 0.7158 | 97 | 这是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了的话第一点就是那功能不足 | cer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0005749601 | 0.7132 | 101 | 这是我们刚才说的那个五点啊就五点配套的在稍微把它总结一下了的话第一点就是那功能不足 | cer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0005781428 | 0.7644 | 39 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0005781428 | 0.7277 | 43 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0005781428 | 0.7173 | 45 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0005781428 | 0.7277 | 41 | 作为一个科研把它做了一个新产品 | cer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0005818033 | 0.7135 | 43 | 投票一个那个教程的啊给咱们这边 | cer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0005818033 | 0.7193 | 43 | 投稿一个那个教程的啊给咱们这边 | cer | 0.1333 |
| 640 | streaming_asr | emilia_zh_0005818033 | 0.7018 | 43 | 投稿一个那个教程的啊给咱们这边 | cer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0005818033 | 0.7193 | 43 | 投稿一个那个教程啊给咱们这边 | cer | 0.0667 |
| 160 | streaming_asr | emilia_zh_0005818215 | 0.8342 | 61 | 却打起去情绪气啊还比较低然后在加上我们要让最近都挺多事儿 | cer | 0.3548 |
| 320 | streaming_asr | emilia_zh_0005818215 | 0.8144 | 69 | 觉得大家其实确实情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.1935 |
| 640 | streaming_asr | emilia_zh_0005818215 | 0.8094 | 67 | 就大家请去情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.2581 |
| 1280 | streaming_asr | emilia_zh_0005818215 | 0.8094 | 66 | 就大家请去情绪气啊还比较低然后在加上我们要要最近都挺多事儿 | cer | 0.2581 |
| 160 | streaming_asr | emilia_zh_0005853036 | 0.7397 | 77 | 这个这个就是如果我想找的话我不排斥啊价值经济介绍的也不是没有想 | cer | 0.5312 |
| 320 | streaming_asr | emilia_zh_0005853036 | 0.7079 | 83 | 作为这个就是如果我想找的话我不排斥啊家里亲戚介绍的人不是没有想 | cer | 0.4062 |
| 640 | streaming_asr | emilia_zh_0005853036 | 0.7270 | 76 | 有这个就如果我想找的话我不排斥啊家里亲戚介绍的人不是没想 | cer | 0.3438 |
| 1280 | streaming_asr | emilia_zh_0005853036 | 0.7175 | 80 | 呃这个就如果我想找的话我不排斥啊家里亲戚介绍的人不是没想 | cer | 0.3438 |
| 160 | streaming_asr | emilia_zh_0005903796 | 0.6844 | 94 | 而且刚才我咱们录节目之前我朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0976 |
| 320 | streaming_asr | emilia_zh_0005903796 | 0.6531 | 102 | 而且刚才我咱们录节目之前我有朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0732 |
| 640 | streaming_asr | emilia_zh_0005903796 | 0.6406 | 106 | 而且刚才我咱们录节目之前我有朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0732 |
| 1280 | streaming_asr | emilia_zh_0005903796 | 0.6406 | 102 | 而且刚才我咱们录节目之前我有朋友给我发个信说他正在打车然后打车的司机大姐跟他说 | cer | 0.0732 |
| 160 | streaming_asr | emilia_zh_0005905391 | 0.8095 | 39 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 320 | streaming_asr | emilia_zh_0005905391 | 0.7965 | 42 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 640 | streaming_asr | emilia_zh_0005905391 | 0.7965 | 41 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 1280 | streaming_asr | emilia_zh_0005905391 | 0.8009 | 40 | 呃这影响的感觉是什么呢就是麦当劳 | cer | 0.1875 |
| 160 | streaming_asr | emilia_zh_0005928718 | 0.7964 | 53 | 很好还是一直关着呢船长现在是他重要的一个牌 | cer | 0.1364 |
| 320 | streaming_asr | emilia_zh_0005928718 | 0.7891 | 54 | 船行还是一直关着呢船长现在是他重要的一个牌 | cer | 0.0909 |
| 640 | streaming_asr | emilia_zh_0005928718 | 0.7782 | 55 | 船行还是一只关着呢船长现在是他重要的一个牌 | cer | 0.1364 |
| 1280 | streaming_asr | emilia_zh_0005928718 | 0.7818 | 55 | 船行还是一个关着嗯船长现在是他重要的一个牌 | cer | 0.1364 |
| 160 | streaming_asr | emilia_zh_0005999475 | 0.7179 | 111 | 哦在生活上你没有感觉到特别的贵因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.2353 |
| 320 | streaming_asr | emilia_zh_0005999475 | 0.7110 | 117 | 然后在生活上你没有感觉到特别的贵因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.2745 |
| 640 | streaming_asr | emilia_zh_0005999475 | 0.6927 | 125 | 然后在生活上你没有感觉到特别的鬼因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0005999475 | 0.6904 | 127 | 然后在生活上因为没有感觉到特别的鬼因为我听说今年全球的音费是之下然后新加坡现在变得异常的鬼 | cer | 0.3137 |
| 160 | streaming_asr | emilia_zh_0006000255 | 0.6667 | 93 | 因为上次我在旅行天天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.1081 |
| 320 | streaming_asr | emilia_zh_0006000255 | 0.6408 | 97 | 因为上次我在旅行天天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.1081 |
| 640 | streaming_asr | emilia_zh_0006000255 | 0.6278 | 102 | 因为上次我在旅行天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.0811 |
| 1280 | streaming_asr | emilia_zh_0006000255 | 0.6343 | 99 | 因为上次我在旅行天给大家介绍的是一个德国的路线嘛然后我觉得两位小姐妹也跟我说 | cer | 0.0811 |
| 160 | streaming_asr | emilia_zh_0006041629 | 0.6931 | 105 | 就看了一些信息以后觉得虽然是够的一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.1458 |
| 320 | streaming_asr | emilia_zh_0006041629 | 0.6955 | 111 | 所以看了一下一些信息以后觉得虽然觉得歌词的一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0006041629 | 0.6832 | 115 | 就是看完这些信息以后觉得虽然觉得够的一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.1042 |
| 1280 | streaming_asr | emilia_zh_0006041629 | 0.6807 | 116 | 就是看完这些信息以后就既然这歌词了一些创伤吧但是整体来说我觉得这个演员他在我眼里变得特别有趣 | cer | 0.1458 |
| 160 | streaming_asr | emilia_zh_0006056256 | 0.7580 | 51 | 我问啊这这种情况就特别容易发生因为我觉得完全 | cer | 0.2727 |
| 320 | streaming_asr | emilia_zh_0006056256 | 0.7534 | 50 | 那啊就这这种情况就特别容易发生因为我觉得完全 | cer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0006056256 | 0.7397 | 53 | 那啊啊这这种情况就特别容易发行因为我觉得完全 | cer | 0.2727 |
| 1280 | streaming_asr | emilia_zh_0006056256 | 0.7671 | 48 | 我问啊这这种情况就特别容易发行因为我觉得完全 | cer | 0.3182 |
| 160 | streaming_asr | emilia_zh_0006099379 | 0.8100 | 47 | 啊等了你一次卷卷到这个量然后他分开使用 | cer | 0.2105 |
| 320 | streaming_asr | emilia_zh_0006099379 | 0.8029 | 50 | 啊等了你一次捐捐到这个量然后他发分开使用 | cer | 0.1579 |
| 640 | streaming_asr | emilia_zh_0006099379 | 0.7885 | 51 | 哦懂了你一次捐捐到这个量然后他把分开使用 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0006099379 | 0.7885 | 54 | 哦懂了你一次捐捐到这个量然后他他分开使用 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0006119067 | 0.7177 | 50 | 而且一点还有一个很让人人就是想的明白的点这个是 | cer | 0.2174 |
| 320 | streaming_asr | emilia_zh_0006119067 | 0.6938 | 54 | 而且也得还有一个很让人人就是想的明白的点啊这个是是 | cer | 0.2174 |
| 640 | streaming_asr | emilia_zh_0006119067 | 0.6699 | 58 | 而且也还有一个很让人人就是想我们明白的点啊这个是是 | cer | 0.2609 |
| 1280 | streaming_asr | emilia_zh_0006119067 | 0.6507 | 61 | 而且也还有一个很让人就是想我们明白的点啊这个是是 | cer | 0.2174 |
| 160 | streaming_asr | emilia_zh_0006119250 | 0.8328 | 49 | 以这种呢我本来啊天蓬说蓬啊蓬马蓬马然后还着点事儿 | cer | 0.5172 |
| 320 | streaming_asr | emilia_zh_0006119250 | 0.7934 | 60 | 有人说呢我本来啊天蓬说我蓬啊蓬马蓬马然后还着点事儿 | cer | 0.4828 |
| 640 | streaming_asr | emilia_zh_0006119250 | 0.7738 | 63 | 一个就是呢我本来啊天平说我朋友我捧马捧马然后还着点事儿 | cer | 0.5172 |
| 1280 | streaming_asr | emilia_zh_0006119250 | 0.7672 | 64 | 意思就是说呢我本来啊天蓬说我蓬吧捧马捧马然后还着点事儿 | cer | 0.3793 |
| 160 | streaming_asr | emilia_zh_0006174150 | 0.7669 | 56 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006174150 | 0.7632 | 59 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006174150 | 0.7519 | 62 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006174150 | 0.7444 | 62 | 就是因为我的一些当时不好的情绪其实会有传染的 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006212201 | 0.7005 | 56 | 那没有办法一直坚持别说的举动所以买通了一个仆人 | cer | 0.1739 |
| 320 | streaming_asr | emilia_zh_0006212201 | 0.6667 | 61 | 他没有办法抑制监视别墅的举动所以买通了一个仆人 | cer | 0.0870 |
| 640 | streaming_asr | emilia_zh_0006212201 | 0.6618 | 59 | 他没有办法一直坚持别说的举动所以买通了一个仆人 | cer | 0.1304 |
| 1280 | streaming_asr | emilia_zh_0006212201 | 0.6522 | 61 | 他没有办法抑制监视别墅的举动所以买通了一个仆人 | cer | 0.0870 |
| 160 | streaming_asr | emilia_zh_0006212326 | 0.8220 | 104 | 嗯远远这个个真正是这小老虎只是他一直在冒着追逐了个上的王子隐藏着自己的真实身份 | cer | 0.4000 |
| 320 | streaming_asr | emilia_zh_0006212326 | 0.8074 | 106 | 嗯远远这个个真正是这小老虎就是他一直在冒着追逐了个上的王子隐藏的自己的真实身份 | cer | 0.4250 |
| 640 | streaming_asr | emilia_zh_0006212326 | 0.8091 | 106 | 路远远这个个真正是这小老虎只是他一直带着帽子伸出了个头上的王子隐藏的自己的真实身份 | cer | 0.3250 |
| 1280 | streaming_asr | emilia_zh_0006212326 | 0.8155 | 99 | 如原来这个个真正是这小老虎只是他一直在冒着你这出的恶作剧上的王子隐藏的自己的真实身份 | cer | 0.4000 |
| 160 | streaming_asr | emilia_zh_0006270122 | 0.7537 | 63 | 你这样不是不对的啊还是一个完整的这个原因啊它就是对应 | cer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0006270122 | 0.7388 | 65 | 你这样都是不对的啊是一个完整的这个原因啊它就是恶意 | cer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0006270122 | 0.7090 | 67 | 你这样不是不对的啊他是一个完整的这个原因啊他就是恶意 | cer | 0.2400 |
| 1280 | streaming_asr | emilia_zh_0006270122 | 0.7052 | 69 | 你这样就是不对的啊他是一个完整的这个原因啊他就是恶意 | cer | 0.2400 |
| 160 | streaming_asr | emilia_zh_0006303057 | 0.7786 | 179 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那个无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你只要吗外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.1216 |
| 320 | streaming_asr | emilia_zh_0006303057 | 0.7712 | 187 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你知道吗外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.1081 |
| 640 | streaming_asr | emilia_zh_0006303057 | 0.7650 | 193 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那个无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你知道吗外圈的玻璃做升降机在外面就是在那个窗户外面 | cer | 0.0676 |
| 1280 | streaming_asr | emilia_zh_0006303057 | 0.7660 | 187 | 一人吗就是以前那个无限挑战的时候他们是一起的然后那无限挑战有一期干什么呢庞明素擦这个外圈的玻璃你知道吗外圈的玻璃做生降级在外面就是在那个窗户外面 | cer | 0.1081 |
| 160 | streaming_asr | emilia_zh_0006304027 | 0.7147 | 94 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应的的轮音是哪一个呢 | cer | 0.1143 |
| 320 | streaming_asr | emilia_zh_0006304027 | 0.7003 | 99 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应到的罗马音是哪一个呢 | cer | 0.0286 |
| 640 | streaming_asr | emilia_zh_0006304027 | 0.6830 | 100 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应到的轮轮是哪一个呢 | cer | 0.0857 |
| 1280 | streaming_asr | emilia_zh_0006304027 | 0.6945 | 96 | 如果按照我们刚刚所讲的这个规律来看的话这里我们对应到的轮轮是哪一个呢 | cer | 0.0857 |
| 160 | streaming_asr | emilia_zh_0006330534 | 0.7320 | 56 | I had no answer for several days At last I received a short note | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006330534 | 0.7285 | 56 | I had no answer for several days at last I received a short note | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006330534 | 0.7113 | 58 | I had no answer for several days at last I received a short note | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006330534 | 0.7285 | 55 | I had no answer for several days at last I received a short note | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006350396 | 0.6835 | 75 | My aunt found the opposition as a clock with spenlo and drew gongs and we found logins nearby | wer | 0.2941 |
| 320 | streaming_asr | emilia_zh_0006350396 | 0.6519 | 80 | My aunt found me a position as a clock with spenlo and your jogins and we found logins nearby | wer | 0.3529 |
| 640 | streaming_asr | emilia_zh_0006350396 | 0.6329 | 79 | My aunt found the opposition as a clock with spenlo and drew juggins and we found logins nearby | wer | 0.2941 |
| 1280 | streaming_asr | emilia_zh_0006350396 | 0.6234 | 78 | My aunt found the opposition as a clock with spenlo and drew juggins and we found logins nearby | wer | 0.2941 |
| 160 | streaming_asr | emilia_zh_0006350510 | 0.6187 | 73 | John said they have lived a very isolated life in India and he was terribly shy of women | wer | 0.1111 |
| 320 | streaming_asr | emilia_zh_0006350510 | 0.6115 | 77 | Draw said the had lived a very isolated life in India and he was terribly shy of women | wer | 0.1111 |
| 640 | streaming_asr | emilia_zh_0006350510 | 0.6043 | 79 | John said they had lived a very isolated life in India and he was terribly shy of women | wer | 0.0556 |
| 1280 | streaming_asr | emilia_zh_0006350510 | 0.6115 | 79 | John said the had lifted very isolated life in India and he was terribly shy of women | wer | 0.1667 |
| 160 | streaming_asr | emilia_zh_0006366492 | 0.7156 | 103 | He did not want to sit in God his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0385 |
| 320 | streaming_asr | emilia_zh_0006366492 | 0.7062 | 102 | He didn't not want to sit and God his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0385 |
| 640 | streaming_asr | emilia_zh_0006366492 | 0.6949 | 109 | He didn't not want to sit and guard his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0769 |
| 1280 | streaming_asr | emilia_zh_0006366492 | 0.7156 | 105 | He didn't not want to sit and guard his great treasure He wanted to return and live with men These riches will give me great power | wer | 0.0769 |
| 160 | streaming_asr | emilia_zh_0006366864 | 0.7277 | 41 | I wanted to continue to make bleak house I happy home for him | wer | 0.0769 |
| 320 | streaming_asr | emilia_zh_0006366864 | 0.7098 | 44 | I wanted to continue to make bleak house I happy home for him | wer | 0.0769 |
| 640 | streaming_asr | emilia_zh_0006366864 | 0.6652 | 46 | I wanted to continue to make bleak house a happy home for him | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006366864 | 0.6652 | 47 | I wanted to continue to make bleak house a happy home for him | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006379722 | 0.7294 | 56 | But he couldn't leave the other to two He must take them with him all the way | wer | 0.0625 |
| 320 | streaming_asr | emilia_zh_0006379722 | 0.7261 | 57 | But he couldn't leave the other to two He must take them with him all the way | wer | 0.0625 |
| 640 | streaming_asr | emilia_zh_0006379722 | 0.6931 | 60 | But he couldn't leave the other to two He must take them with him all the way | wer | 0.0625 |
| 1280 | streaming_asr | emilia_zh_0006379722 | 0.7228 | 57 | But he couldn't leave the other of two He must take them with him all the way | wer | 0.0625 |
| 160 | streaming_asr | emilia_zh_0006379861 | 0.6379 | 77 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006379861 | 0.6310 | 74 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006379861 | 0.6034 | 80 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006379861 | 0.6034 | 79 | When the professor went into the cell he had one five dollar bill and two ten dollar bills | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006430098 | 0.7003 | 56 | John was offered several good jobs But he wanted to wait and look around | wer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006430098 | 0.6873 | 63 | John was offered several good jobs but he wanted to wait and look around | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006430098 | 0.6873 | 62 | John was offered several good jobs but he wanted to wait and look around | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006430098 | 0.6873 | 64 | John was offered several good jobs but he wanted to wait and look around | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006430973 | 0.6273 | 92 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 320 | streaming_asr | emilia_zh_0006430973 | 0.6025 | 92 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 640 | streaming_asr | emilia_zh_0006430973 | 0.5994 | 93 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 1280 | streaming_asr | emilia_zh_0006430973 | 0.5807 | 96 | The chest of George was something that Gregor could do without if he had to but the writing desk had to stay | wer | 0.1818 |
| 160 | streaming_asr | emilia_zh_0006435274 | 0.7080 | 44 | I bank charges interest a brother doesn't charge interest | wer | 0.1111 |
| 320 | streaming_asr | emilia_zh_0006435274 | 0.6726 | 46 | A bank charges interest a brother doesn't charge interest | wer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006435274 | 0.6991 | 44 | A bank charges interest a brother doesn't charge interest | wer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006435274 | 0.6858 | 47 | A bank charges interest a brother doesn't charge interest | wer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006435497 | 0.6579 | 190 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in middle you will see see and here in a preciation of what end log again is AKA merch sort today | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0006435497 | 0.6327 | 200 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in the middle you will see see and here in a appreciation of what end log again is AKA merch sort today | wer | 0.1556 |
| 640 | streaming_asr | emilia_zh_0006435497 | 0.6289 | 194 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubbles sort and in the middle you will see and here in a appreciation of what end log again is AKA merch sort today | wer | 0.1333 |
| 1280 | streaming_asr | emilia_zh_0006435497 | 0.6264 | 192 | The middle in the bottom three different algorithms on the top you will see selection sort on the bottom you will see bubble sort and in the middle you will see and here in a appreciation of what end log again is AKA merch sort today | wer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0006446583 | 0.6676 | 92 | Now we can use emori emaging to see what is actually happening inside the joint When someone cracks nuckles | wer | 0.2000 |
| 320 | streaming_asr | emilia_zh_0006446583 | 0.6511 | 93 | Now we can use emori emaging to see what is actually happening inside the joint when someone cracks nuckles | wer | 0.2000 |
| 640 | streaming_asr | emilia_zh_0006446583 | 0.6456 | 93 | Now we can use emori emaging to see what is actually happening inside the joint when someone cracks the nuckles | wer | 0.2000 |
| 1280 | streaming_asr | emilia_zh_0006446583 | 0.6236 | 96 | Now we can use MRI imaging to see what is actually happening inside the joint when someone cracks the nuckles | wer | 0.1000 |
| 160 | streaming_asr | emilia_zh_0006447049 | 0.5772 | 189 | You know part of him thanks like what what look can I contribute to society and my like good enough Do my friends love me and I are wrote a mother very quick note say Just as what I take to realize after I turned 30 not too long ago | wer | 0.2800 |
| 320 | streaming_asr | emilia_zh_0006447049 | 0.5772 | 190 | You know part of him thinks like what else can I contribute the society and my like good enough Do my friends love me and I are wrote a very good quick note say this is a what I pay to realize after I turn 30 not through long ago | wer | 0.3200 |
| 640 | streaming_asr | emilia_zh_0006447049 | 0.5561 | 191 | You know part of him thinks like what else can I contribute to society and my good enough Do my friends love me and I are wrote a very good quick note to say this is a what I pay to realize after I turn 30 not too long ago | wer | 0.2600 |
| 1280 | streaming_asr | emilia_zh_0006447049 | 0.5593 | 190 | You know part of him thinks like what else can I contribute to society and my right good enough Do my friends love me and I are wrote a very very quick note say This is a what I take to realize after I turn 30 not too long ago | wer | 0.3000 |
| 160 | streaming_asr | emilia_zh_0006464698 | 0.6463 | 113 | I'll law to for his actually for example he wanted me to tell him my big account pastword and he kept taking photos of me from different angles | wer | 0.2963 |
| 320 | streaming_asr | emilia_zh_0006464698 | 0.6092 | 121 | A lot of ways I actually for example he wanted me to tell him my big account pastword and he kept taking photos of me from different angles | wer | 0.1481 |
| 640 | streaming_asr | emilia_zh_0006464698 | 0.6026 | 120 | A lot of for his actually for example he wanted me to tell him my bank account pastword and he kept taking photos of me from different angles | wer | 0.1481 |
| 1280 | streaming_asr | emilia_zh_0006464698 | 0.6004 | 121 | A longer of ways actually for example he wanted me to tell him my bank account pastword and he kept taking photos of me from different angles | wer | 0.1111 |
| 160 | streaming_asr | emilia_zh_0006464935 | 0.6520 | 119 | That price the point where the quality that consumers want to buy a equal the quality that sellers want to produce is called the equal librarian price | wer | 0.1538 |
| 320 | streaming_asr | emilia_zh_0006464935 | 0.6226 | 121 | That price the point were the quantity that consumers want to buy a equal the quantity that sellers want to produce is called the equal librarian price | wer | 0.2308 |
| 640 | streaming_asr | emilia_zh_0006464935 | 0.6143 | 122 | That price the point were the quantity that consumers want to buy a equal the quantity that sellers want to produce is called the equal librarian price | wer | 0.2308 |
| 1280 | streaming_asr | emilia_zh_0006464935 | 0.6164 | 119 | That price the point were the quantity that consumers want to buy a equal the quantity that sellers want to produce is called the equal librarian price | wer | 0.2308 |
| 160 | streaming_asr | emilia_zh_0006503627 | 0.7975 | 89 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的烧杀抢掠后被摧毁殆尽成为一片废墟 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0006503627 | 0.7871 | 93 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的稍稍抢日后被摧毁殆尽成为一片废墟 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0006503627 | 0.7766 | 99 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的稍稍抢日后被摧毁殆尽成为一片废墟 | cer | 0.1389 |
| 1280 | streaming_asr | emilia_zh_0006503627 | 0.7766 | 98 | 昔日的繁华都会情怀盛敬经历如此肆无忌惮的稍稍抢日后被摧毁殆尽成为一片废墟 | cer | 0.1389 |
| 160 | streaming_asr | emilia_zh_0006544664 | 0.7789 | 118 | 后院货运商贸走了很勤这种小车在广州那么晚的都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1702 |
| 320 | streaming_asr | emilia_zh_0006544664 | 0.7705 | 125 | 货运货运商贸走了很勤这种小车在广州那么短的都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1489 |
| 640 | streaming_asr | emilia_zh_0006544664 | 0.7739 | 128 | 贺运霍玉商贸走了很勤这种小车在广州那满大街都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1277 |
| 1280 | streaming_asr | emilia_zh_0006544664 | 0.7638 | 131 | 客运货运商贸走了很勤这种小车在广州那么满的都是而且司机当时也没注意只注意到车的颜色是那种灰啥了 | cer | 0.1277 |
| 160 | streaming_asr | emilia_zh_0006544707 | 0.7447 | 90 | 这两个人呢也是在你说的这个区间有冬季结婚的现象而且这两个人其中有一个 | cer | 0.0588 |
| 320 | streaming_asr | emilia_zh_0006544707 | 0.7342 | 90 | 这两个人呢也是在你说的这个区间有冬季结婚的现象而且这两个人其中有一个 | cer | 0.0588 |
| 640 | streaming_asr | emilia_zh_0006544707 | 0.7263 | 92 | 这两个人呢也是在你说这个区间有冬季结婚的现象而且这两个呢人其中有一个 | cer | 0.1176 |
| 1280 | streaming_asr | emilia_zh_0006544707 | 0.7158 | 94 | 这两个人呢也是在你说这个区间有冬季结婚的现象而且这两个呢人其中有一个 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0006598681 | 0.7033 | 82 | 就是在公牛大学学习半年之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0303 |
| 320 | streaming_asr | emilia_zh_0006598681 | 0.6800 | 87 | 就是在公有大学学习半年之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006598681 | 0.6700 | 89 | 就是在公有大学学习半年之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006598681 | 0.6867 | 86 | 就是在公有大学学习半点之后第一份给你安排的工作直接就是到博物馆了吗 | cer | 0.0303 |
| 160 | streaming_asr | emilia_zh_0006610357 | 0.6667 | 114 | 今天我觉得是非常确实关键性是非常之前但是似乎欧盟的这种做法它其实也并没有什么错我觉得 | cer | 0.1628 |
| 320 | streaming_asr | emilia_zh_0006610357 | 0.6800 | 112 | 之前我觉得是非常确实关键性是非常之前但是似乎恶魔的这种做法它其实也并没有什么错我觉得 | cer | 0.1628 |
| 640 | streaming_asr | emilia_zh_0006610357 | 0.6747 | 110 | 最近我觉得是非常确实关键性是非常之前但是似乎欧盟的这种做法它其实也并没有什么错我觉得 | cer | 0.1628 |
| 1280 | streaming_asr | emilia_zh_0006610357 | 0.6747 | 111 | 这件我觉得是非常确实关键性是非常之前但是似乎欧盟的这种做法它其实也并没有什么错我觉得 | cer | 0.1395 |
| 160 | streaming_asr | emilia_zh_0006658157 | 0.7292 | 127 | 所以我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这一个产品本身它在 | cer | 0.0600 |
| 320 | streaming_asr | emilia_zh_0006658157 | 0.7254 | 126 | 所以我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这个产品本身它在 | cer | 0.0800 |
| 640 | streaming_asr | emilia_zh_0006658157 | 0.7197 | 129 | 所以我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这个产品本身它在 | cer | 0.0800 |
| 1280 | streaming_asr | emilia_zh_0006658157 | 0.7140 | 131 | 对我觉得是这样子说所以任何品牌在中国办活动它肯定也是因为这一个活动本身或者说就这个产品本身它在 | cer | 0.1200 |
| 160 | streaming_asr | emilia_zh_0006713619 | 0.6743 | 88 | 不管啊那些新人们你们要互关赶紧互关起来再给你们三分钟时间互关玩下播了啊 | cer | 0.1351 |
| 320 | streaming_asr | emilia_zh_0006713619 | 0.6612 | 94 | 不管啊那些新人们你们要互关赶紧互关起来再给你们三分钟时间互关玩下播了啊 | cer | 0.1351 |
| 640 | streaming_asr | emilia_zh_0006713619 | 0.6513 | 94 | 不管啊啊那些新人们你们要护光赶紧护光起来再给你们三分钟时间护光我要下播了啊 | cer | 0.2703 |
| 1280 | streaming_asr | emilia_zh_0006713619 | 0.6447 | 94 | 不管啊那些新人们你们要护光赶紧护光起来再给你们三分钟时间护光我要下播了啊 | cer | 0.2432 |
| 160 | streaming_asr | emilia_zh_0006714517 | 0.6726 | 50 | 好接下来我讲最重要的今天的一件事情呢啊 | cer | 0.1000 |
| 320 | streaming_asr | emilia_zh_0006714517 | 0.6548 | 52 | 好接下来我要讲最重要的今天的一件事情呢啊 | cer | 0.0500 |
| 640 | streaming_asr | emilia_zh_0006714517 | 0.6488 | 52 | 好接下来我要讲最重要的今天的一件事情呢啊 | cer | 0.0500 |
| 1280 | streaming_asr | emilia_zh_0006714517 | 0.6488 | 54 | 好接下来我要讲最重要的今天的一件事情呢啊 | cer | 0.0500 |
| 160 | streaming_asr | emilia_zh_0006725267 | 0.6770 | 48 | 老狼猪小刘蛋也是能帮助部分人逃脱怨闷 | cer | 0.5789 |
| 320 | streaming_asr | emilia_zh_0006725267 | 0.6522 | 50 | 伯兰登住小刘旦也只能帮助部分人逃出院门 | cer | 0.3684 |
| 640 | streaming_asr | emilia_zh_0006725267 | 0.6584 | 52 | 伯大人出小刘旦也使得帮助部分人逃出院门 | cer | 0.4211 |
| 1280 | streaming_asr | emilia_zh_0006725267 | 0.6522 | 51 | 伯大人住小刘旦也只能帮助部分人逃出院门 | cer | 0.3684 |
| 160 | streaming_asr | emilia_zh_0006731464 | 0.7056 | 48 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006731464 | 0.7056 | 48 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0006731464 | 0.6889 | 49 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006731464 | 0.6889 | 49 | 他选择这种生活方式一定有他自己的道理 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0006819602 | 0.7655 | 81 | 有半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0645 |
| 320 | streaming_asr | emilia_zh_0006819602 | 0.7628 | 83 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0968 |
| 640 | streaming_asr | emilia_zh_0006819602 | 0.7628 | 84 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0968 |
| 1280 | streaming_asr | emilia_zh_0006819602 | 0.7574 | 85 | 又半个时辰过去城墙的高度还剩下不到一张曹军完全停下了田土的动作 | cer | 0.0968 |
| 160 | streaming_asr | emilia_zh_0006874664 | 0.7152 | 121 | 那其中他有说到他非常诚恳的就是希望如果说是金水怎么样如果是查查怎么样但对于这一段描述里面有觉得比较真实 | cer | 0.1154 |
| 320 | streaming_asr | emilia_zh_0006874664 | 0.6814 | 138 | 那其中他有说到他非常诚恳的就是希望如果说是金水怎么样如果是查查怎么样那对于这一段描述里面有觉得比较真实 | cer | 0.1154 |
| 640 | streaming_asr | emilia_zh_0006874664 | 0.6962 | 134 | 那其中它有说到它非常诚恳的就是希望如果说是金水怎么样如果是查杀怎么样那对这一段描述里面有觉得比较真实 | cer | 0.1154 |
| 1280 | streaming_asr | emilia_zh_0006874664 | 0.6920 | 135 | 那其中他有说到他非常诚恳的就是希望如果说是金水怎么样如果是查杀怎么样那对这一段描述里面有觉得比较真实 | cer | 0.0769 |
| 160 | streaming_asr | emilia_zh_0006883085 | 0.7594 | 48 | 所以与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.1579 |
| 320 | streaming_asr | emilia_zh_0006883085 | 0.7453 | 51 | 在与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.0526 |
| 640 | streaming_asr | emilia_zh_0006883085 | 0.7358 | 51 | 在与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.0526 |
| 1280 | streaming_asr | emilia_zh_0006883085 | 0.7311 | 55 | 在与终于停下之际我都已经有点昏昏欲睡了 | cer | 0.0526 |
| 160 | streaming_asr | emilia_zh_0006940399 | 0.8134 | 45 | 虽然他们爱人都想尽气前嫌从跌倒的地方爬起来 | cer | 0.0952 |
| 320 | streaming_asr | emilia_zh_0006940399 | 0.8172 | 46 | 虽然他们爱人都想尽气前嫌从跌倒的地方爬起来 | cer | 0.0952 |
| 640 | streaming_asr | emilia_zh_0006940399 | 0.7836 | 54 | 虽然他们二人都想进去浅显从跌倒的地方爬起来 | cer | 0.1905 |
| 1280 | streaming_asr | emilia_zh_0006940399 | 0.7799 | 55 | 虽然他们二人都想进去浅显从跌倒的地方爬起来 | cer | 0.1905 |
| 160 | streaming_asr | emilia_zh_0006940509 | 0.7586 | 38 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 320 | streaming_asr | emilia_zh_0006940509 | 0.7241 | 43 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 640 | streaming_asr | emilia_zh_0006940509 | 0.7299 | 43 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 1280 | streaming_asr | emilia_zh_0006940509 | 0.7241 | 43 | 宏伟都市的<\|write_generate\|><\|cmn\|><\|start_content\|>街道很快便不满了红云 | cer | 2.8667 |
| 160 | streaming_asr | emilia_zh_0006990963 | 0.7902 | 41 | 现在正好是春天我们一会儿就去院子里 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0006990963 | 0.7723 | 44 | 现在正好是春天我们一块儿就去院子里 | cer | 0.0588 |
| 640 | streaming_asr | emilia_zh_0006990963 | 0.7768 | 43 | 现在正好是春天我们一会儿就去院子里 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0006990963 | 0.7768 | 45 | 现在正好是春天我们一会儿就去院子里 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007017041 | 0.7742 | 67 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007017041 | 0.7683 | 69 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007017041 | 0.7742 | 70 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007017041 | 0.7742 | 70 | 几乎是百分之九十以上的地方政府包括州和县都无法完成预算 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007017174 | 0.7410 | 83 | 从某种程度上啊当然不是出特地演个的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0909 |
| 320 | streaming_asr | emilia_zh_0007017174 | 0.7300 | 88 | 从某种程度上啊当然不是出特别严格的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007017174 | 0.7245 | 91 | 从某种程度上啊当人不是出特别严格的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0303 |
| 1280 | streaming_asr | emilia_zh_0007017174 | 0.7245 | 89 | 从某种程度上啊当然不是出特别严格的说法啊就是所得税恐怕就是工薪税啊 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007060532 | 0.7500 | 46 | 如果你对某个人偶见是五十真爱就别用嘴说 | cer | 0.2632 |
| 320 | streaming_asr | emilia_zh_0007060532 | 0.7308 | 49 | 如果你对某个人偶见是五十真爱就不用嘴说 | cer | 0.3158 |
| 640 | streaming_asr | emilia_zh_0007060532 | 0.7212 | 49 | 如果你对某个人偶见是五十真爱就别用嘴说 | cer | 0.2632 |
| 1280 | streaming_asr | emilia_zh_0007060532 | 0.7356 | 50 | 如果你对某个人偶见是五十真爱就别用嘴说 | cer | 0.2632 |
| 160 | streaming_asr | emilia_zh_0007120510 | 0.7653 | 41 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007120510 | 0.7551 | 43 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007120510 | 0.7449 | 45 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007120510 | 0.7500 | 45 | 他将极大的激励整个团队伙伴的信心 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007121845 | 0.7923 | 108 | 那么我们会也许只是依赖于一个技术比较方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0488 |
| 320 | streaming_asr | emilia_zh_0007121845 | 0.7871 | 110 | 那么我们会也许只是依赖于一个技术比一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0244 |
| 640 | streaming_asr | emilia_zh_0007121845 | 0.7784 | 112 | 那么我们会也许只是依赖于一个技术一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007121845 | 0.7766 | 113 | 那么我们会也许只是依赖于一个技术比一个方法一个优势或者是资源一些能力我们就去对抗市场 | cer | 0.0244 |
| 160 | streaming_asr | emilia_zh_0007124543 | 0.7827 | 74 | 印度所受的恐吓风险在一期中最好印度尼西亚也曾经面临高风险 | cer | 0.1071 |
| 320 | streaming_asr | emilia_zh_0007124543 | 0.7775 | 73 | 印度所述的空隙风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0007124543 | 0.7853 | 71 | 印度所受的恐袭风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.0357 |
| 1280 | streaming_asr | emilia_zh_0007124543 | 0.7827 | 72 | 印度所受的恐袭风险在一期中最高印度尼西亚也曾经面临高风险 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0007124790 | 0.8220 | 57 | 你好请问这是去老人之家的路吧男人看了小军一眼 | cer | 0.0455 |
| 320 | streaming_asr | emilia_zh_0007124790 | 0.8164 | 55 | 你好请问这是去老人之家的路吧男人看了小军一眼 | cer | 0.0455 |
| 640 | streaming_asr | emilia_zh_0007124790 | 0.8192 | 55 | 你好请问这是去老人之家的路吧男人看了小俊一眼 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007124790 | 0.8192 | 52 | 你好请问这是去老人之家的路吧男人看了小俊一眼 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007169925 | 0.8336 | 109 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包的他们也会利用任何时间思考 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007169925 | 0.8252 | 111 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 640 | streaming_asr | emilia_zh_0007169925 | 0.8210 | 114 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包等他们也会利用任何时间思考 | cer | 0.0238 |
| 1280 | streaming_asr | emilia_zh_0007169925 | 0.8182 | 117 | 而犹太人似乎在这方面要更胜一筹因为在犹太人里面即使是烤面包蛋他们也会利用任何时间思考 | cer | 0.0238 |
| 160 | streaming_asr | emilia_zh_0007170191 | 0.7419 | 41 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007170191 | 0.7366 | 40 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007170191 | 0.7473 | 41 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007170191 | 0.7473 | 42 | 这对犹太人来说肯定是不能接受的 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007312342 | 0.8101 | 82 | 凯勒他人生中的第一个支票账户他将他怎么写支票一天上班时丹尼斯提到 | cer | 0.0938 |
| 320 | streaming_asr | emilia_zh_0007312342 | 0.8059 | 86 | 凯勒他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0938 |
| 640 | streaming_asr | emilia_zh_0007312342 | 0.8101 | 86 | 开了他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0312 |
| 1280 | streaming_asr | emilia_zh_0007312342 | 0.8038 | 82 | 开了他人生中的第一个支票账户他交他怎么写支票一天上班时丹尼斯提到 | cer | 0.0312 |
| 160 | streaming_asr | emilia_zh_0007312674 | 0.7566 | 109 | 事实上亨利也不需要你亲爱的他根本不知道自己身在何地也认不出身边是谁和他待在一起 | cer | 0.0256 |
| 320 | streaming_asr | emilia_zh_0007312674 | 0.7586 | 105 | 事实上亨利也不需要你亲爱的他根本不知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007312674 | 0.7566 | 109 | 事实上通力也不需要你亲爱的的他根本不知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0769 |
| 1280 | streaming_asr | emilia_zh_0007312674 | 0.7627 | 106 | 事实上亨利也不需要你亲爱的他根本不知道自己身在何地也认不出身边是谁和他呆在一起 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007353483 | 0.8192 | 53 | 普通的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.2273 |
| 320 | streaming_asr | emilia_zh_0007353483 | 0.8222 | 57 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 640 | streaming_asr | emilia_zh_0007353483 | 0.8251 | 57 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 1280 | streaming_asr | emilia_zh_0007353483 | 0.8222 | 58 | 不同的选择会创造出不同的未来关键词二虚拟利益 | cer | 0.1364 |
| 160 | streaming_asr | emilia_zh_0007399682 | 0.7788 | 91 | 有些上市甚至一个更加严厉的态度仅我们要认清生命的脆弱告诉我们每一个人 | cer | 0.1471 |
| 320 | streaming_asr | emilia_zh_0007399682 | 0.7677 | 90 | 有些丧失甚至一个更加严厉的态度仅是我们要认清生命的脆弱告诉我们每一个人 | cer | 0.1765 |
| 640 | streaming_asr | emilia_zh_0007399682 | 0.7699 | 89 | 有些丧失甚至以更加严厉的态度警示我们要认清生命的脆弱告诉我们每一个人 | cer | 0.0588 |
| 1280 | streaming_asr | emilia_zh_0007399682 | 0.7699 | 91 | 有些丧失甚至以更加严厉的态度竟是我们要认清生命的脆弱告诉我们每一个人 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0007399687 | 0.7076 | 45 | 他在一千多年前许下的诺言知识不需 | cer | 0.1875 |
| 320 | streaming_asr | emilia_zh_0007399687 | 0.7135 | 44 | 他在一千多年前许下的落言知识不需 | cer | 0.2500 |
| 640 | streaming_asr | emilia_zh_0007399687 | 0.7193 | 44 | 他在一千多年前许下的落言真实不需 | cer | 0.1250 |
| 1280 | streaming_asr | emilia_zh_0007399687 | 0.7193 | 43 | 他在一千多年前许下的落言真实不需 | cer | 0.1250 |
| 160 | streaming_asr | emilia_zh_0007461662 | 0.8102 | 85 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 320 | streaming_asr | emilia_zh_0007461662 | 0.8000 | 89 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 640 | streaming_asr | emilia_zh_0007461662 | 0.8041 | 88 | 把那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 1280 | streaming_asr | emilia_zh_0007461662 | 0.8020 | 86 | 吧那么这个是导师感兴趣的领域啊第二个是自己感兴趣的领域我想啊这个是大家嗯 | cer | 0.0278 |
| 160 | streaming_asr | emilia_zh_0007462823 | 0.7650 | 48 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 320 | streaming_asr | emilia_zh_0007462823 | 0.7650 | 47 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 640 | streaming_asr | emilia_zh_0007462823 | 0.7373 | 54 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 1280 | streaming_asr | emilia_zh_0007462823 | 0.7281 | 55 | 因为他说到picnic所以我们的图片a是picnic的 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0007524507 | 0.7723 | 47 | 好我们在来看用骰子走的这个就是送去更改它的颜色 | cer | 0.3913 |
| 320 | streaming_asr | emilia_zh_0007524507 | 0.7589 | 50 | 好我们在来看用下载的这个搜索我们去更改它的颜色 | cer | 0.2174 |
| 640 | streaming_asr | emilia_zh_0007524507 | 0.7188 | 57 | 好我们在来看用下角的这个这个送去更改它的颜色 | cer | 0.2609 |
| 1280 | streaming_asr | emilia_zh_0007524507 | 0.7098 | 59 | 好我们在来看用下角的这个这个我们去更改它的颜色 | cer | 0.1739 |
| 160 | streaming_asr | emilia_zh_0007526333 | 0.7268 | 99 | 哦了这些各的最重点一个就是每年考试都会考这个内容就是普希拉戏剧作家和作品 | cer | 0.1667 |
| 320 | streaming_asr | emilia_zh_0007526333 | 0.7191 | 102 | 到了这些各的最重点一个就是每年考试都会考这个内容就是普希拉戏剧作家和作品 | cer | 0.1389 |
| 640 | streaming_asr | emilia_zh_0007526333 | 0.7242 | 101 | 到了这些各的最重点一个就是每年考试都会考这个内容就是古希腊戏剧作家和作品 | cer | 0.0833 |
| 1280 | streaming_asr | emilia_zh_0007526333 | 0.7294 | 98 | 到了这些各的最重点一个就是每年考试都会考这个内容就是古希腊戏剧作家和作品 | cer | 0.0833 |
| 160 | streaming_asr | emilia_zh_0007551053 | 0.7346 | 73 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 320 | streaming_asr | emilia_zh_0007551053 | 0.7443 | 74 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 640 | streaming_asr | emilia_zh_0007551053 | 0.7411 | 75 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 1280 | streaming_asr | emilia_zh_0007551053 | 0.7411 | 74 | 其他更多的工作是由律师事务所的有工作经验的前台前辈们进行了 | cer | 0.0357 |
| 160 | streaming_asr | emilia_zh_0007555536 | 0.7370 | 67 | 其你将其余的三个房间里的收集全部版过来我要细细的查阅一下 | cer | 0.1786 |
| 320 | streaming_asr | emilia_zh_0007555536 | 0.7163 | 75 | 请你将其余的三个房间里的收集全部版过来我要细细的查阅一下 | cer | 0.1429 |
| 640 | streaming_asr | emilia_zh_0007555536 | 0.7093 | 75 | 请你将其余的三个房间里的收集全部把过来我要细细的查阅一下 | cer | 0.1429 |
| 1280 | streaming_asr | emilia_zh_0007555536 | 0.6955 | 77 | 请你将其余的三个房间里的收集全部把过来我要细细的查阅一下 | cer | 0.1429 |
| 160 | streaming_asr | emilia_zh_0007635379 | 0.7071 | 53 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 320 | streaming_asr | emilia_zh_0007635379 | 0.7222 | 46 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 640 | streaming_asr | emilia_zh_0007635379 | 0.6717 | 54 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 1280 | streaming_asr | emilia_zh_0007635379 | 0.6818 | 53 | 所以来比是对于德国人来说是一个很重要的城市 | cer | 0.0952 |
| 160 | streaming_asr | emilia_zh_0007635686 | 0.7925 | 60 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 320 | streaming_asr | emilia_zh_0007635686 | 0.7862 | 57 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007635686 | 0.7925 | 56 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007635686 | 0.7799 | 62 | 我们的工作仍然没有实质性进展呢抓了几个小特务 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007690753 | 0.7627 | 129 | 可能市场每次新想到的是仅仅过来一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.1765 |
| 320 | streaming_asr | emilia_zh_0007690753 | 0.7593 | 132 | 可能是场每次生响到的时仅仅过来一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.2353 |
| 640 | streaming_asr | emilia_zh_0007690753 | 0.7559 | 133 | 可能是场没成想到的是仅仅过了一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.1373 |
| 1280 | streaming_asr | emilia_zh_0007690753 | 0.7559 | 135 | 可能是场没曾想到的是仅仅过了一年搞了二零零四年最漫长的下跌再度发生上证终止再次出现越线五连运的悲惨场面 | cer | 0.1176 |
| 160 | streaming_asr | emilia_zh_0007691054 | 0.7547 | 95 | 把从这个角度来讲我们认为allie是一个比定律论增速更好更加重要的指标原因很简单 | cer | 0.2432 |
| 320 | streaming_asr | emilia_zh_0007691054 | 0.7547 | 97 | 那从这个角度来讲我们认为ROE是一个比定律论增速更好更加重要的指标远很简单 | cer | 0.1892 |
| 640 | streaming_asr | emilia_zh_0007691054 | 0.7500 | 98 | 那从这个角度来讲我们认为all是一个比定律论增速更好更加重要的指标源于很简单 | cer | 0.2162 |
| 1280 | streaming_asr | emilia_zh_0007691054 | 0.7570 | 96 | 那从这个角度来讲我们认为ROE是一个比定律论增速更好更加重要的指标远远很简单 | cer | 0.1892 |
| 160 | streaming_asr | emilia_zh_0007721270 | 0.7019 | 46 | 当我接管家公司时我通常会做两件事儿 | cer | 0.0556 |
| 320 | streaming_asr | emilia_zh_0007721270 | 0.7019 | 45 | 当我接管一家公司时我通常会做两件事儿 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007721270 | 0.7081 | 44 | 当我接管一家公司时我通常会做两件事儿 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007721270 | 0.7019 | 44 | 当我接管一家公司时我通常会做两件事儿 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007721307 | 0.7425 | 91 | 其实马云的很多演讲里呢也会出现某些主题反复讲在现象马云说呢重复是为了强调 | cer | 0.0278 |
| 320 | streaming_asr | emilia_zh_0007721307 | 0.7400 | 93 | 其实马云的很多演讲里呢也会出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007721307 | 0.7425 | 92 | 其实马云的很多演讲里呢也会出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007721307 | 0.7250 | 97 | 其实马云的很多演讲里呢也会出现某些主题反复讲的现象马云说呢重复是为了强调 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007761299 | 0.6764 | 78 | 我觉得这个行业不适合我我觉得我跟这个咱公司的同事处的不开心 | cer | 0.0357 |
| 320 | streaming_asr | emilia_zh_0007761299 | 0.6727 | 78 | 我觉得这个行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0000 |
| 640 | streaming_asr | emilia_zh_0007761299 | 0.6582 | 80 | 我觉得这个行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0000 |
| 1280 | streaming_asr | emilia_zh_0007761299 | 0.6545 | 80 | 我觉得这个行业不适合我我觉得我跟这咱公司的同事处的不开心 | cer | 0.0000 |
| 160 | streaming_asr | emilia_zh_0007788542 | 0.7517 | 66 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 320 | streaming_asr | emilia_zh_0007788542 | 0.7483 | 64 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 640 | streaming_asr | emilia_zh_0007788542 | 0.7343 | 66 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 1280 | streaming_asr | emilia_zh_0007788542 | 0.7343 | 67 | 不少人开始接触期货觉得比较难以理解什么是期货呢 | cer | 0.0417 |
| 160 | streaming_asr | emilia_zh_0007790200 | 0.7813 | 111 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |
| 320 | streaming_asr | emilia_zh_0007790200 | 0.7690 | 121 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |
| 640 | streaming_asr | emilia_zh_0007790200 | 0.7672 | 121 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |
| 1280 | streaming_asr | emilia_zh_0007790200 | 0.7584 | 123 | 特别指出了现在是小型私营企业卖场的最佳时机大型有实力的<\|write_generate\|><\|cmn\|><\|start_content\|>企业可以考虑向海外牵肠问向海外搬迁 | cer | 1.0000 |

结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。
