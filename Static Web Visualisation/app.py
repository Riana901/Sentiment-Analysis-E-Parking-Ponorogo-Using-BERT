from flask import Flask, render_template, request
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__)

df = pd.read_csv('dataset.csv')
df_train = pd.read_csv('df_train.csv')
df_test = pd.read_csv('df_val.csv')
df_pred = pd.read_csv('data_with_predictions.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=False)
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

try:
    model.load_state_dict(torch.load('sentiment_model.pth', map_location=torch.device('cpu')))
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8  
    )
    quantized_model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")


@app.route('/')
def dashboard():
    try:

        tables = df.to_html(classes='table table-striped', index=False)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        tables = "<p>Error loading data</p>"

    return render_template('dashboard.html', tables=tables)


@app.route('/data_train')
def data_train_page():
    try:
        tables = df_train.to_html(classes='table table-striped', index=False)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        tables = "<p>Error loading data</p>"

    return render_template('data_train.html', tables=tables)

@app.route('/data_test')
def data_test_page():
    try:
        
        tables = df_test.to_html(classes='table table-striped', index=False)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        tables = "<p>Error loading data</p>"

    return render_template('data_test.html', tables=tables)


@app.route('/klasifikasi_bert', methods=['GET', 'POST'])
def klasifikasi_bert_page():
    sentiment = None  
    input_text = None
    if request.method == 'POST':
        print("POST request received")
        input_text = request.form.get('input_text')
        print(f"Received input text: {input_text}")

        if not input_text or input_text.strip() == "":
            sentiment = "Please enter a valid text."
        else:
            try:
                
                inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)

               
                with torch.no_grad():
                    outputs = quantized_model(**inputs)
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits, dim=-1).item()

                if predicted_class == 0:
                    sentiment = 'Negative'
                elif predicted_class == 1:
                    sentiment = 'Neutral'
                else:
                    sentiment = 'Positive'
            except Exception as e:
                sentiment = f"Error: {str(e)}"

    print(f"Sentiment analysis result: {sentiment}")

    try:
    
        tables = df_pred.to_html(classes='table table-striped', index=False)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        tables = "<p>Error loading data</p>"

    model_accuracy = 98

    return render_template('klasifikasi_bert.html', sentiment=sentiment, input_text=input_text, tables=tables, accuracy=model_accuracy)



@app.route('/visualisasi')
def visualisasi_page():
    model_accuracy = 98
    return render_template('visualisasi.html', accuracy=model_accuracy)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
