from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentData(BaseModel):
    Ma_HV: str
    Diem_GK: float
    Diem_ThiThu: float

@app.post("/api/phan-luong")
def phan_luong(data: list[StudentData]):
    df = pd.DataFrame([item.dict() for item in data])
    
    # Thuật toán K-Means
    X = df[['Diem_GK', 'Diem_ThiThu']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Nhom_KMeans'] = kmeans.fit_predict(X_scaled).tolist()
    
    return {"status": "success", "data": df.to_dict(orient="records")}
