from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama
from datetime import datetime
import os

app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


embeddings = download_hugging_face_embeddings()
index_name = "medical-knowledge-base-index"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOllama(
    model="llama3",
    temperature=0.7
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


fallback_responses = {
    "symptoms": "I understand you're experiencing symptoms. Please describe them in detail, including when they started and their severity. Remember, I'm here to provide information, but you should consult a healthcare professional for proper diagnosis.",
    "fever": "Fever can be a sign of various conditions. Monitor your temperature, stay hydrated, and rest. If fever persists above 101°F (38.3°C) for more than 3 days or is accompanied by severe symptoms, please consult a doctor.",
    "headache": "Headaches can have various causes including stress, dehydration, or underlying conditions. Try rest, hydration, and over-the-counter pain relievers if appropriate. Seek medical attention for severe, sudden, or persistent headaches.",
    "i have fever": "Fever can be a sign of various conditions. Monitor your temperature, stay hydrated, and rest. If fever persists above 101°F (38.3°C) for more than 3 days or is accompanied by severe symptoms, please consult a doctor.",
    "i have a headache": "Headaches can have various causes including stress, dehydration, or underlying conditions. Try rest, hydration, and over-the-counter pain relievers if appropriate. Seek medical attention for severe, sudden, or persistent headaches.",
    "first aid tips": "Here are basic first aid tips: 1) For cuts - clean and bandage, 2) For burns - cool water for 10-20 minutes, 3) For choking - perform Heimlich maneuver, 4) Always call emergency services for serious injuries.",
    "emergency": "🚨 For medical emergencies, please call 911 immediately or go to your nearest emergency room. This chatbot is not for emergency situations.",
    "health tips": "General health tips: 1) Stay hydrated (8 glasses water/day), 2) Exercise regularly (30 min/day), 3) Eat balanced diet with fruits/vegetables, 4) Get 7-9 hours sleep, 5) Regular checkups with your doctor.",
    "default": "I'm here to help with your health questions. Please provide more specific information about your symptoms or concerns. Remember, this is for informational purposes only and doesn't replace professional medical advice."
}

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        
        if request.is_json:
            user_message = request.json.get('message', '').strip()
        else:
            
            user_message = request.form.get("msg", "").strip()
        
        if not user_message:
            return jsonify({
                'response': "Please enter a message.",
                'timestamp': datetime.now().strftime('%H:%M')
            })
        
        print(f"User input: {user_message}")
        
    
        emergency_keywords = ['emergency', 'urgent', 'dying', 'heart attack', 'stroke', 'suicide', 'overdose']
        if any(keyword in user_message.lower() for keyword in emergency_keywords):
            return jsonify({
                'response': fallback_responses["emergency"],
                'timestamp': datetime.now().strftime('%H:%M')
            })
        
   
        try:
            response = rag_chain.invoke({"input": user_message})
            ai_response = response["answer"]
            
           
            if len(ai_response.strip()) > 20 and "I'm not sure based on the available information" not in ai_response:
                print("RAG Response:", ai_response)
                return jsonify({
                    'response': ai_response,
                    'timestamp': datetime.now().strftime('%H:%M')
                })
            else:
               
                raise Exception("RAG response not sufficient")
                
        except Exception as rag_error:
            print(f"RAG chain error or insufficient response: {rag_error}")
            
           
            user_lower = user_message.lower()
            ai_response = fallback_responses["default"]
            
          
            if user_lower in fallback_responses:
                ai_response = fallback_responses[user_lower]
            else:
             
                for keyword, reply in fallback_responses.items():
                    if keyword in user_lower and keyword != "default":
                        ai_response = reply
                        break
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().strftime('%H:%M')
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'error': 'Sorry, I encountered an error. Please try again.',
            'timestamp': datetime.now().strftime('%H:%M')
        }), 500


@app.route("/get", methods=["GET", "POST"])
def get_response():
    try:
        msg = request.form["msg"]
        user_input = msg
        print(user_input)
        response = rag_chain.invoke({"input": user_input})
        print("Response : ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print(f"Get endpoint error: {e}")
        return "Sorry, I encountered an error processing your request."

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

@app.route('/health-tips')
def health_tips():
    return render_template('health_tips.html')


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'llama3',
        'vector_store': 'pinecone',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)