import time
import random

def postprocess_response(response):
    end_token = None
    
    if '\n' in response:
        response = response.replace('\n', ' ')
    if ':' in response:
        response = response.split(': ')[-1]
    response = response.strip()
        
    if response.endswith('[변화 대화]'):
        response = response.split('[변화 대화]')[0].strip()
        end_token = 'change talk'
        
    elif response.endswith('[종료]'):
        response = response.split('[종료]')[0].strip()
        end_token = 'terminate'
    
    return response, end_token

# prob의 확률로 True return
def random_choice(prob):
    current_time_seed = int(time.time())
    random.seed(current_time_seed)
    return random.random() < prob

# String의 첫 글자 대문자로
def capitalize(s):
    if not s:
        return s  # Return the original string if it's empty
    return s[0].upper() + s[1:]

# Dialogue history를 T5 모델의 input 형태로 변환
def history_to_t5_input(messages_en, tag_history, window_size=6, include_label='OnlyTherapist'):
    
    # T5 모델이 학습한 label명과 동일하게 변환
    convert_speaker = {'user': 'client', 'assistant': 'therapist'}
    convert_label = {
        'Simple Reflection': 'Simple Reflection',
        'Complex Reflection': 'Complex Reflection',
        'Open Question': 'Open Question',
        'Closed Question': 'Closed Question',
        'Affirm': 'Affirm',
        'Give Information': 'Give Information',
        'Advise': 'Advise', # T5 모델은 'Advise'로 학습함
        'Other': 'Other',
        
        'Change Talk': 'Change',
        None: 'Not Change'
    }
    
    t5_input = f"Predict next therapist's dialogue act: "
    
    # 마지막 window_size개만큼의 history만 남기기
    if len(messages_en) > window_size:
        messages_en = messages_en[-window_size:]
        tag_history = tag_history[-window_size:]

    # Therapist, client의 label 둘 다 표기
    if include_label == True:
        for utterance, label in zip(messages_en, tag_history):
            t5_input += f"[{capitalize(convert_speaker[utterance['role']])}: "
            t5_input += f"{convert_label[label]}] "
            t5_input += utterance['content']
            t5_input += ' '

    # Therapist의 label만 표기
    elif include_label == 'OnlyTherapist':
        for utterance, label in zip(messages_en, tag_history):
            if utterance['role'] == 'assistant':
                t5_input += f"[{capitalize(convert_speaker[utterance['role']])}: "
                t5_input += f"{convert_label[label]}] "
            else:
                t5_input += f"[{capitalize(convert_speaker[utterance['role']])}] "
            t5_input += utterance['content']
            t5_input += ' '

    # Label 표기 x
    else:
        for utterance, label in zip(messages_en, tag_history):
            t5_input += f"[{capitalize(convert_speaker[utterance['role']])}] "
            t5_input += utterance['content']
            t5_input += ' '

    return t5_input.strip()

# top_k_pred 예시 : ['[Therapist: Give Information]', '[Therapist: Other]', '[Therapist: Open Question]']
# output: ['Give Information', 'Other', 'Open Question']
def pred_parser(top_k_pred):
    label_candidates = ['Simple Reflection', 'Complex Reflection', 'Open Question', 'Closed Question', 'Affirm', 'Give Information', 'Advise', 'Other']
    labels = []

    # Parsing이 안 되는 response에 label명이 포함되어 있으면 그 label로 살리기 (대소문자 구분 x)
    def find_label_from_wrong_response(text):
        for label in [l.lower() for l in label_candidates]:
            if label in text.lower():
                return label.title()
        return None

    for i, pred in enumerate(top_k_pred):
        try:
            label = pred[1:-1].split(': ')[1] # ex. 'Give Information'
            if label in label_candidates:
                labels.append(label)
            else:
                raise Exception('try second_parsing')
            
        except:
            second_parsing = find_label_from_wrong_response(pred)
            if second_parsing != None:
                labels.append(second_parsing)
            else:
                print(f'Prediction Parsing Error (Top-{i+1}): {pred}')
                labels.append(None)
                
    return labels

# input으로 들어온 prediction이 rule에 걸릴 경우 None return
# prediction: 하나의 tag만 (ex. 'Simple Reflection')
def final_tag_choice(prediction, counselor_tag_history, cut_oq, last_turn=False):

    if prediction == None:
        return None

    # Open Question일 경우 cut_oq의 확률로 None return (OQ가 너무 많이 나와서 줄일 장치)
    elif prediction == 'Open Question' and random_choice(cut_oq):
        # print('\n확률적으로 OQ 거절')
        return None

    # 대화의 마지막 turn에는 질문 x
    elif last_turn and prediction.split(' ')[-1] == 'Question':
        return None
    
    else:
        if len(counselor_tag_history) >= 2:
            
            # 똑같은 tag 연속으로 3번 x
            if prediction == counselor_tag_history[-1] == counselor_tag_history[-2]:
                return None
            
            # open이든 closed든 질문 연속으로 3번 x
            elif prediction.split(' ')[-1] == counselor_tag_history[-1].split(' ')[-1] == counselor_tag_history[-2].split(' ')[-1] == 'Question':
                return None
            
            # simple이든 complex든 반영 연속으로 4번 x
            elif len(counselor_tag_history) >= 3 and prediction.split(' ')[-1] == counselor_tag_history[-1].split(' ')[-1] == counselor_tag_history[-2].split(' ')[-1] == counselor_tag_history[-3].split(' ')[-1] == 'Reflection':
                return None
            
            else:
                return prediction
            
        else:
            return prediction

# MI Forecaster가 예측한 k개의 tag 중 사용할 하나의 tag return
# k개의 tag 중 중복 있어도 상관 없음
# 가능한 게 하나도 없으면 휴리스틱하게 선택
# top_k_prediction: MI Forecaster가 예측한 k개의 tag (ex. ['Simple Reflection', 'Open Question', None])
def final_tag_choice_from_top_k(top_k_prediction, tag_history, cut_oq, last_turn=False, print_choice=False):
    counselor_tag_history = tag_history[0::2]

    # Top-2 Prediction 안에 Advise 있으면 무조건 Advise 선택 (Advise 늘릴 방법)
    if 'Advise' in top_k_prediction[:2]:
        return 'Advise'

    for p in top_k_prediction:
        final_tag = final_tag_choice(p, counselor_tag_history, cut_oq, last_turn)
        if final_tag != None:
            return final_tag

    # Global Seed 42에서 다시 랜덤하게
    current_time_seed = int(time.time())
    random.seed(current_time_seed)

    # 상담사가 이전에 2번 연속으로 질문을 한 경우 - 2가지 반영 중에서 랜덤 선택
    # Update: SR이 CR보다 적게 나와서 무조건 SR로
    if counselor_tag_history[-1].split(' ')[-1] == counselor_tag_history[-2].split(' ')[-1] == 'Question':
        random_reflection = random.choice(['Simple Reflection'])
        if print_choice:
            print(f'Heuristic Choice: {random_reflection}')
        return random_reflection
    
    # 상담사가 이전에 2번 연속으로 반영을 한 경우 - 2가지 질문 중에서 랜덤 선택
    elif counselor_tag_history[-1].split(' ')[-1] == counselor_tag_history[-2].split(' ')[-1] == 'Reflection':
        random_question = random.choice(['Open Question', 'Closed Question'])
        if print_choice:
            print(f'Heuristic Choice: {random_question}')
        return random_question
    
    # 나머지 경우 - Simple Reflection, Complex Reflection, Other 중에서 랜덤 선택
    # Update: SR이 CR보다 적게 나와서 Simple Reflection, Other 중에서 랜덤 선택
    else:
        random_selection = random.choice(['Simple Reflection', 'Other'])
        if print_choice:
            print(f'Heuristic Choice: {random_selection}')
        return random_selection