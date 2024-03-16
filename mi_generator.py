from openai import OpenAI
import openai, json
from simplet5 import SimpleT5
import pandas as pd
from tqdm import tqdm
import random
import time
import json
from key_info import OPENAI_API_KEY, UPSTAGE_API_KEY
from mi_database import counselor
from utils import *

# API Key & Variables
chat_model = 'gpt-4-0125-preview' # gpt-4-0125-preview, gpt-4-1106-preview, gpt-3.5-turbo
client = OpenAI(api_key = OPENAI_API_KEY)
client_ko2en = OpenAI(
  api_key=UPSTAGE_API_KEY,
  base_url="https://api.upstage.ai/v1/solar"
)

# MI Forecaster Model
model = SimpleT5()
model.load_model("t5","mi_forecaster_model", use_gpu=True)

already_done = 0 # 생성되는 첫 번째 대화의 id: already_done+1
quantity = 200 # 생성할 대화 개수, None이면 전체 생성
min_turns = 8 # 최소 2*min_turns-1 턴 이후에 대화 종료 ([종료] 토큰에 의해)
max_turns = 16 # 2*max_turns-1 턴 도달하면 대화 종료
cut_oq = 0.2 # OQ가 나왔을 때 랜덤하게 거절할 확률 (OQ가 너무 많이 나와서 줄일 장치)
affirm_prob = 0.2 # 내담자의 change talk 이후에 상담사가 affirm을 하게 할 확률
full_dialogues = []
full_tags = []
error_report = {}

category = '정신건강' # ['정신건강', '대인관계', '자아·성격', '취업·진로', '학업·고시', '중독·집착', '가족']
context_description_file_path = f'seed_data/mindcafe_score_2_{category}.json'
save_dialogue_path = f'result/KMI_{category}_id{already_done+1}~.json'
error_report_path = f'result/KMI_{category}_id{already_done+1}~_error.json'

automatic_tag_selection = True # False할 경우 print_progress는 True로
print_turn = False
print_prompt = False
print_progress = False

convert_speaker = {'user': '내담자', 'assistant': '상담사'}
abbr = {
    'Simple Reflection': 'SR',
    'Complex Reflection': 'CR',
    'Open Question': 'OQ',
    'Closed Question': 'CQ',
    'Affirm': 'AF',
    'Give Information': 'GI',
    'Advise': 'AD',
    'Other': 'OT' 
}

# Load the context description
with open(context_description_file_path, 'r') as f:
    context_description = json.load(f)

# Define the functions
def ko2en(ko):
    time.sleep(1.5)
    stream = client_ko2en.chat.completions.create(
    model="solar-1-mini-translate-koen",
    messages=[
        {
        "role": "user",
        "content": ko
        }
    ],
    stream=False,
    )
    
    return stream.choices[0].message.content.strip()

def counselor_simulator(dialogue_history, messages_en, tag_history, turn, min_turn, max_turn):
    terminate = False
    if print_turn:
        print(f'\nTurn (Counselor): {turn}/{max_turn}\n')

    # 첫번째 턴
    if turn == 1:
        prompt = '''
당신은 상담사(counselor)입니다. 아래 예시와 같이 처음 상담을 하러 온 내담자(client)에게 할 열린 질문(Open Question)을 생성해주세요. 이 때, 아래의 제약조건을 지켜주세요.

제약조건:
- 존댓말을 사용해야 합니다.
- 1문장으로 작성해야 합니다.
- 줄바꿈(newline)은 넣으면 안 됩니다.
- 한 문장은 한 절(cluase)로만 구성해주세요. '~며' 등의 연결어를 사용하지 말아주세요.
- '첫걸음', '한 걸음', '첫 단계', '한 단계', '작은 변화', '시작점', '지지', '지원', '자원', '당신', '귀하', '여러분'과 같은 단어는 절대 사용하면 안 됩니다.
- '매우'와 같은 극단적인 표현은 사용하지 말아주세요.
- 상대방을 칭할 때 '내담자님'이라고 말해주세요.

예시:
상담사: 안녕하세요, 어떤 일로 상담을 받으러 오셨나요?
상담사: 안녕하세요, 오늘 어떤 것에 대해 같이 이야기를 나눠볼까요?
상담사: 안녕하세요, 어떤 고민으로 찾아오게 되셨나요?

상담사:
'''
        if print_prompt:
            print('-'*40)
            print('[Counselor Simulator] Prompt:')
            print(prompt)
            print('-'*40)

        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        ).choices[0].message.content

        response, _ = postprocess_response(response)

        return response.strip(), 'Other', terminate # 원래는 Open Question, 다 생성된 다음에 Open Question으로 바꿔줌

    # 첫번째 이후의 턴
    else:
        if tag_history[-1] == 'Change Talk' and random_choice(affirm_prob):
            if print_progress:
                print('\n확률적으로 Affirm 선택')
            tag = 'AF'
        else:
            # MI Forecaster Inference
            t5_input = history_to_t5_input(messages_en, tag_history)
            # print(t5_input)
            prediction = pred_parser(model.predict(t5_input, num_beams=10, num_return_sequences=3))               
            final_prediction = final_tag_choice_from_top_k(prediction, tag_history, cut_oq, last_turn=(turn==max_turn), print_choice=True)
            if print_progress:
                print(f'MI Forecaster Prediction: {prediction}') # prediction 리스트에 동일한 원소가 여러 개 들어있을 수도 있음
                print(f'MI Forecaster Final Prediction: {final_prediction}')

            if automatic_tag_selection:
                tag = abbr[final_prediction]
            else:
                tag = input('\t*[SR, CR, OQ, CQ, AF, GI, AD, OT] 중 하나의 태그를 선택해주세요: ')
                if tag not in ['SR', 'CR', 'OQ', 'CQ', 'AF', 'GI', 'AD', 'OT']:
                    tag = input('\t*[SR, CR, OQ, CQ, AF, GI, AD, OT] 중 하나의 태그를 다시 선택해주세요: ')

        if tag == 'OT':
            initial_prompt = '''
당신은 상담사(counselor)입니다. 아래의 제약조건을 지키고, 주어진 대화 다음에 올 상담사(counselor)의 발화(utterance)를 생성해주세요. 대화를 통해 내담자가 현재 상황에서 변화하고자 하는 내적 동기를 이끌어내는 것이 이 상담의 목적입니다.

제약조건:
- 존댓말을 사용해야 합니다.
- 절대 질문은 하면 안 됩니다.
- 답변은 1문장 또는 2문장으로 작성해야 합니다.
- 답변에 줄바꿈(newline)은 넣으면 안 됩니다.
- 한 문장은 한 절(cluase)로만 구성해주세요. '~며' 등의 연결어를 사용하지 말아주세요.
- '첫걸음', '한 걸음', '첫 단계', '한 단계', '작은 변화', '시작점', '지지', '지원', '자원', '당신', '귀하', '여러분'과 같은 단어는 절대 사용하면 안 됩니다.
- '매우'와 같은 극단적인 표현은 사용하지 말아주세요.
- 상대방을 칭할 때 '내담자님'이라고 말해주세요.
'''
            if turn >= min_turn:
                initial_prompt += "- 내담자의 고민이 어느 정도 해결되어서 상담을 마무리지어도 될 것 같다면 문장의 마지막에 '[종료]'를 붙여주세요.\n"
            initial_prompt += '\n'

            if counselor[tag]['ex'] is not None:
                for i, e in enumerate(counselor[tag]['ex']):
                    initial_prompt += f"예시 {i+1}:\n" # 한국어
                    for u in e:
                        initial_prompt += u + '\n'
                    initial_prompt += '\n'

            prompt = initial_prompt + '대화:\n' + dialogue_history + f"상담사: "

        # 질문일 때는 [종료] 붙이라는 프롬프트 x
        elif tag in ['OQ', 'CQ']:
            initial_prompt = f'''
당신은 상담사(counselor)입니다. 아래의 제약조건을 지키고, 주어진 대화 다음에 올 상담사(counselor)의 발화(utterance)를 만드세요. 대화를 통해 내담자가 현재 상황에서 변화하고자 하는 내적 동기를 이끌어내는 것이 이 상담의 목적입니다. 이 때, '{counselor[tag]['kor']}({counselor[tag]['eng']})'에 근거해서 생성해주세요.

제약조건:
- 존댓말을 사용해야 합니다.
- 답변은 1문장 또는 2문장으로 작성해야 합니다.
- 답변에 줄바꿈(newline)은 넣으면 안 됩니다.
- 한 문장은 한 절(cluase)로만 구성해주세요. '~며' 등의 연결어를 사용하지 말아주세요.
- '첫걸음', '한 걸음', '첫 단계', '한 단계', '작은 변화', '시작점', '지지', '지원', '자원', '당신', '귀하', '여러분'과 같은 단어는 절대 사용하면 안 됩니다.
- '매우'와 같은 극단적인 표현은 사용하지 말아주세요.
- 상대방을 칭할 때 '내담자님'이라고 말해주세요.

'''
            initial_prompt += f"'{counselor[tag]['kor']}'의 정의:\n" # 한국어
            for d in counselor[tag]['def']:
                initial_prompt += d + '\n'
            initial_prompt += '\n'

            if counselor[tag]['ex'] is not None:
                for i, e in enumerate(counselor[tag]['ex']):
                    initial_prompt += f"'{counselor[tag]['kor']}'의 예시 {i+1}:\n" # 한국어
                    for u in e:
                        initial_prompt += u + '\n'
                    initial_prompt += '\n'

            prompt = initial_prompt + '대화:\n' + dialogue_history + f"상담사 [{counselor[tag]['kor']}]: "

        elif tag in ['SR', 'CR', 'AF', 'GI', 'AD']:
            initial_prompt = f'''
당신은 상담사(counselor)입니다. 아래의 제약조건을 지키고, 주어진 대화 다음에 올 상담사(counselor)의 발화(utterance)를 만드세요. 대화를 통해 내담자가 현재 상황에서 변화하고자 하는 내적 동기를 이끌어내는 것이 이 상담의 목적입니다. 이 때, '{counselor[tag]['kor']}({counselor[tag]['eng']})'에 근거해서 생성해주세요.

제약조건:
- 존댓말을 사용해야 합니다.
- 답변은 1문장 또는 2문장으로 작성해야 합니다.
- 답변에 줄바꿈(newline)은 넣으면 안 됩니다.
- 한 문장은 한 절(cluase)로만 구성해주세요. '~며' 등의 연결어를 사용하지 말아주세요.
- '첫걸음', '한 걸음', '첫 단계', '한 단계', '작은 변화', '시작점', '지지', '지원', '자원', '당신', '귀하', '여러분'과 같은 단어는 절대 사용하면 안 됩니다.
- '매우'와 같은 극단적인 표현은 사용하지 말아주세요.
- 상대방을 칭할 때 '내담자님'이라고 말해주세요.
'''
            if turn >= min_turn:
                initial_prompt += "- 내담자의 고민이 어느 정도 해결되어서 상담을 마무리지어도 될 것 같다면 문장의 마지막에 '[종료]'를 붙여주세요.\n"
            initial_prompt += '\n'

            initial_prompt += f"'{counselor[tag]['kor']}'의 정의:\n" # 한국어
            for d in counselor[tag]['def']:
                initial_prompt += d + '\n'
            initial_prompt += '\n'

            if counselor[tag]['ex'] is not None:
                for i, e in enumerate(counselor[tag]['ex']):
                    initial_prompt += f"'{counselor[tag]['kor']}'의 예시 {i+1}:\n" # 한국어
                    for u in e:
                        initial_prompt += u + '\n'
                    initial_prompt += '\n'

            prompt = initial_prompt + '대화:\n' + dialogue_history + f"상담사 [{counselor[tag]['kor']}]: "

        if print_prompt:
            print('-'*40)
            print('[Counselor Simulator] Prompt:')
            print(prompt)
            print('-'*40)

        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        ).choices[0].message.content

        response, end_token = postprocess_response(response)
        if end_token == 'terminate':
            terminate = True

        return response.strip(), counselor[tag]['eng'], terminate


def counselee_simulator(context, dialogue_history, tag_history, turn, max_turn):
    change = None
    if print_turn:
        print(f'\nTurn (Counselee): {turn}/{max_turn}\n')

    initial_prompt = '''
당신은 아래 상황의 문제로 상담사(counselor)에게 상담하러 간 내담자(client)입니다. 아래의 제약조건을 지키고, 주어진 대화 다음에 올 내담자(client)의 답변(response)을 만드세요. 아래 상황에 대한 고민을 상담사(counselor)에게 천천히, 조금씩 이야기해보세요. 만약 상담사와의 대화로 인해 이 상황을 변화시키고 싶은 의지가 생겼고 대화 맥락 상 변화 대화를 하는 것이 자연스러울 경우 '변화 대화'를 생성해주세요. 변화 대화를 생성할 경우, 예시와 같이 문장의 마지막에 '[변화 대화]'를 붙여주세요.

제약조건:
- 존댓말을 사용해야 합니다.
- 상담사는 아래 상황에 대한 배경지식이 없는 상태로, 상황에 대해 구체적으로 설명해주어야 합니다.
- 주어진 상황을 바탕으로 하되, 구체적인 상황을 자연스럽게 지어내서 추가해도 됩니다.
- 생성하는 내담자의 답변이 이전 대화와 자연스럽게 연결되어야 합니다.
- 답변은 1문장 또는 2문장이어야 합니다.
- 한 문장은 한 절(cluase)로만 구성해주세요. '~며' 등의 연결어를 사용하지 말아주세요.
- '첫걸음', '한 걸음', '첫 단계', '한 단계', '작은 변화', '시작점', '지지', '지원', '자원', '당신', '귀하', '여러분'과 같은 단어는 절대 사용하면 안 됩니다.
- '조언을 듣고 싶습니다'라는 구절은 절대 사용하면 안 됩니다.

'변화 대화'의 정의:
자신의 문제에 대해 스스로 말하기를 변화하고 싶다거나 변화와 관련된 진술 혹은 언어표현이다.
아래와 같은 것들이 변화 대화에 포함된다:
1. 변화에 대한 희망, 변화하고 싶다는 언어적 진술과 표현 (Desire)
2. 변화할 수 있다는 생각, 변화에 대한 낙관적인 시각, 변화는 가능하다 혹은 변화할 것이라는 표현 (Ability)
3. 변화의 이득과 장점, 변화로 인해 긍정적인 결과가 생길 것이라는 표현 (Reason)
4. 변화의 필요성, 변화하지 않을 때의 문제점과 손실, 변화하지 않는 것에 대한 걱정, 염려 및 우려 (Need)

'변화 대화'의 예시 1:
내담자: 운전할 때 제가 평정심을 좀 유지하고 싶어요. [변화 대화]

'변화 대화'의 예시 2:
내담자: 제가 노력하면 이것은 할 수 있습니다. [변화 대화]

'변화 대화'의 예시 3:
내담자: 사람들이랑 더 많은 대화를 나누게 된다면 더 많은 친구들을 사귈 수 있을 거에요. [변화 대화]

'변화 대화'의 예시 4:
내담자: 이렇게 계속 살이 찌면 안 돼요. [변화 대화]
'''

# # 저항 추가
#     initial_prompt = '''
# 당신은 아래 상황의 문제로 상담사(counselor)에게 상담하러 간 내담자(client)입니다. 아래의 제약조건을 지키고, 주어진 대화 다음에 올 내담자(client)의 답변(response)을 만드세요. 아래 상황에 대한 고민을 상담사(counselor)에게 천천히, 조금씩 이야기해보세요. 처음에는 상담사에 대한 저항을 드러내주세요. 그러다가 상담사와 대화를 하면서 저항 정도가 점차 낮아져야 합니다.

# 제약조건:
# - 존댓말을 사용해야 합니다.
# - 자연스러운 문장 구조를 사용해야 합니다.
# - 주어진 상황을 바탕으로 하되, 구체적인 상황을 자연스럽게 지어내서 답변해도 됩니다.
# - 답변을 풍부하게 작성해주세요.
# - 답변은 1문장 또는 2문장이어야 합니다.
# - 한 문장은 한 절(cluase)로만 구성해주세요. '~며' 등의 연결어를 사용하지 말아주세요.
# - '첫걸음', '한 걸음', '첫 단계', '한 단계', '지지', '지원', '당신'과 같은 단어는 절대 사용하면 안 됩니다.
# - '조언을 듣고 싶습니다'라는 구절은 절대 사용하면 안 됩니다.

# 저항의 종류 : 
# (1) 논쟁하기 : 상담사의 말에 정확성, 전문성, 진솔성에 대해 도전하고 깎아내리며 적대감을 표현한다. 
# (2) 방해하기 : 방어적인 태도로 면담자의 말을 가로막거나 자른다. 
# (3) 부인하기 : 자신에게 문제가 없고 면담에 협조하지 않으며 면담자의 조언은 받아들이지 않겠다고 표현한다. 
# (4) 무시하기 : 상담사의 말을 따르지 않거나 무시하며 주의를 기울이지 않거나 엉뚱한 대답을 함.
# '''

    prompt = initial_prompt + f"\n상황:\n{context}" + '\n\n대화:\n' + dialogue_history + f"내담자: "

    if print_prompt:
        print('-'*40)
        print('[Counselee Simulator] Prompt:')
        print(prompt)
        print('-'*40)

    response = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    ).choices[0].message.content

    response, end_token = postprocess_response(response)
    if end_token == 'change talk':
        change = 'Change Talk'

    return response.strip(), change



if __name__ == "__main__":
    start_time = time.time()

    if quantity:
        upper_limit = already_done + quantity # already_done 다음부터 quantity만큼 생성
    else:
        upper_limit = len(context_description) # already_done 다음부터 전체 생성

    for context_idx in tqdm(range(already_done, upper_limit)):
        try:
            topic = context_description[context_idx]['Category']
            context = context_description[context_idx]['Content']

            if print_progress:
                print(f'\nGenerating dialogue {context_idx+1}')
                print('\n[상황]\n',context,'\n')

            messages = [{"topic": topic, "context": context, "dialogue":{'ko':[], 'en':[]}}]
            tags = []
            dialogue_history = ''

            # Start the dialogue (Counselor)
            counselor_response, tag, terminate = counselor_simulator(None, None, None, 1, min_turns, max_turns)
            messages[0]['dialogue']['ko'].append({"role": "assistant", "content": counselor_response})
            messages[0]['dialogue']['en'].append({"role": "assistant", "content": ko2en(counselor_response)})
            
            tags.append(tag)
            dialogue_history += f'상담사: {counselor_response}' + '\n'

            if print_progress:
                for i, j in zip(messages[0]['dialogue']['ko'], tags):
                    print(f"{convert_speaker[i['role']]} [{j}]: {i['content']}")

            # Continue the dialogue
            for turn in range(max_turns-1):
                user_response, tag = counselee_simulator(context, dialogue_history, tags, turn+1, max_turns)
                messages[0]['dialogue']['ko'].append({"role": "user", "content": user_response})
                messages[0]['dialogue']['en'].append({"role": "user", "content": ko2en(user_response)})
                
                tags.append(tag)
                dialogue_history += f'내담자: {user_response}' + '\n'

                if print_progress:
                    print('\n===================================================\n')
                    for i, j in zip(messages[0]['dialogue']['ko'], tags):
                        print(f"{convert_speaker[i['role']]} [{j}]: {i['content']}")

                # Get the counselor response and tag
                counselor_response, tag, terminate = counselor_simulator(dialogue_history, messages[0]['dialogue']['en'], tags, turn+2, min_turns, max_turns)
                messages[0]['dialogue']['ko'].append({"role": "assistant", "content": counselor_response})
                messages[0]['dialogue']['en'].append({"role": "assistant", "content": ko2en(counselor_response)})
                
                tags.append(tag)
                dialogue_history += f'상담사: {counselor_response}' + '\n'

                if print_progress:
                    print('\n===================================================\n')
                    for i, j in zip(messages[0]['dialogue']['ko'], tags):
                        print(f"{convert_speaker[i['role']]} [{j}]: {i['content']}")

                if terminate:
                    if print_progress:
                        print('\n[종료] 토큰으로 상담 종료')
                    break

            full_dialogues.append(messages)
            full_tags.append(tags)
        
            # Post-processing
            final_result = []

            for id, (dialogue, tag) in enumerate(zip(full_dialogues, full_tags)):
                if (dialogue != None) and (tag != None):
                    new_dialogue = []
                    for u_ko, u_en, t in zip(dialogue[0]['dialogue']['ko'], dialogue[0]['dialogue']['en'], tag):
                        new_dialogue.append({'role': 'Client' if u_ko['role']=='user' else 'Therapist', 'utterance_ko': u_ko['content'],'utterance_en': u_en['content'], 'label': 'General' if t=='Other' else t}) #, 'utterance_en': u['en']['content']
                        
                    # 각 대화의 첫 번째 상담사 발화는 Open Question
                    # OQ로 넣으니 MI forecaster 모델이 뒤에서 계속 OQ를 예측해서 Other로 넣은 후 마지막에 OQ로 바꿔주도록 post-processing
                    new_dialogue[0]['label'] = 'Open Question'

                    # 턴 수 계산하는 법 정확히 알
                    final_result.append({'id': id+already_done+1, 'category': dialogue[0]['topic'], 'context': dialogue[0]['context'], 'turns': len(new_dialogue), 'dialogue': new_dialogue})
                    # final_result.append({'id': id+already_done+1, 'category': dialogue[0]['topic'], 'context': dialogue[0]['context'], 'turns': round(len(new_dialogue)/2 + 0.1), 'dialogue': new_dialogue}) # 턴 수 2로 나눔
            
            # Dialogue 하나 생성할 때마다 저장
            with open(save_dialogue_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\nError occurred in id {context_idx+1}: [{str(type(e).__name__)}] {str(e)}")
            error_report[context_idx+1] = f"[{str(type(e).__name__)}] {str(e)}"
            full_dialogues.append(None)
            full_tags.append(None)

            # Error Report 저장 (Error 발생할 때마다)
            with open(error_report_path, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=4)


    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"Runtime: {hours} hours, {minutes} minutes, {seconds} seconds")