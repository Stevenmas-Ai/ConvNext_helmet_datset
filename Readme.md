# 🏗️ Helmet Detection with ConvNeXt 🚧  

A deep learning project using **ConvNeXt** for **helmet detection**, deployed with **FastAPI** and **Streamlit**.  

---

## **📌 Project Overview**

This project trains a **ConvNeXt-based deep learning model** to detect helmets in images.  
It includes:
✅ **Data preprocessing** using **VOC XML annotations**  
✅ **Training a ConvNeXt model** for classification  
✅ **Deployment via FastAPI (backend) & Streamlit (frontend)**  
✅ **Dockerized API & UI for easy deployment**  

---

## **📂 Project Structure**

```bash
CONVNEXT_HELMET_DATASET/
│── helmet_dataset/                  # Dataset folder
│── .gitignore                        # Ignore model files & venv
│── best_model.pth                    # Trained model (not pushed to Git)
│── helmet_detection_convnext.pth           # TorchScript model
│── helmet_detection_model.pt          # Final TorchScript model
│── ConvNeXt_Helmet_dataset.ipynb      # Jupyter Notebook (training script)
│── fastapi_app.py                     # FastAPI backend
│── streamlit_app.py                    # Streamlit frontend
│── docker-compose.yml                  # Docker setup
│── Dockerfile                           # Docker config for FastAPI
│── Dockerfile.streamlit                 # Docker config for Streamlit
│── README.md                            # Project documentation



⸻

🛠️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/Caephas/ConvNext_helmet_datset.git
cd ConvNext_helmet_datset

2️⃣ Set Up Virtual Environment

python3 -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install -r requirements.txt



⸻

📊 Training the Model

1️⃣ Ensure dataset is inside helmet_dataset/
2️⃣ Run the Jupyter Notebook:

jupyter notebook ConvNeXt_Helmet_dataset.ipynb

3️⃣ Train the model and save it:

torch.save(model.state_dict(), "helmet_detection_convnext.pth")

4️⃣ Convert the model to TorchScript for inference:

traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
traced_model.save("helmet_detection_model.pt")



⸻

🚀 Running the FastAPI Backend

uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

✅ API is live at: http://127.0.0.1:8000/docs

⸻

🖥️ Running the Streamlit Frontend

streamlit run streamlit_app.py

✅ UI available at: http://127.0.0.1:8501

⸻

🐳 Running with Docker

1️⃣ Build and Run Containers

docker-compose up --build

✅ FastAPI Backend: http://127.0.0.1:8000
✅ Streamlit Frontend: http://127.0.0.1:8501

⸻

📡 Deployment

Deploy on AWS, Hugging Face, or Docker Hub
 • Push to Docker Hub:

docker tag backend your-dockerhub-username/backend
docker push your-dockerhub-username/backend

 • Deploy on AWS EC2:

sudo docker-compose up --build -d

 • Deploy Streamlit on Streamlit Cloud:
 • Push your repo to GitHub
 • Visit Streamlit Cloud
 • Deploy your repository


 Feel free to contribute! 
