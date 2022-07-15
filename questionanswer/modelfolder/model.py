import torch
from pyexpat import model
from torch.nn.functional import softmax
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained('questionanswer/modelfolder/assets/', local_files_only=True)

        model = AutoModelForQuestionAnswering.from_pretrained('questionanswer/modelfolder/assets/', local_files_only=True)
        self.model = model.to(self.device)

    def predict(self, question, answer_text):
        inputs = self.tokenizer(question, answer_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        answer_start_score = softmax(outputs.start_logits).max()
        answer_end_score = softmax(outputs.end_logits).max()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        # print(tokenizer.decode(predict_answer_tokens))
        # print(answer_start_score, answer_end_score)
        return (
            self.tokenizer.decode(predict_answer_tokens),
            float(answer_start_score*answer_end_score)
        )





# model = AutoModelForQuestionAnswering.from_pretrained('questionanswer/modelfolder/assets/', local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained('questionanswer/modelfolder/assets/', local_files_only=True)



# nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
# QA_input = {
#     'question': 'How many parameters does BERT-large have?',
#     'context': 'BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance.'
# }
# res = nlp(QA_input)
# print(res)

# question, text = 'How many parameters does BERT-large have?', 'BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance.'

# inputs = tokenizer(question, text, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)

# answer_start_index = outputs.start_logits.argmax()
# answer_end_index = outputs.end_logits.argmax()

# answer_start_score = softmax(outputs.start_logits).max()
# answer_end_score = softmax(outputs.end_logits).max()

# predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
# print(tokenizer.decode(predict_answer_tokens))
# print(answer_start_score, answer_end_score)

model = Model()


def get_model():
    return model
