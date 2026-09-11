import sys, os, torch, numpy as np, time
sys.path.insert(0, os.getcwd())
from models.bilstm_branch import BiLSTMForecaster
from models.risk_classifier import RiskClassifier
sys.modules['__main__'].BiLSTMForecaster = BiLSTMForecaster
sys.modules['__main__'].RiskClassifier = RiskClassifier

scale_data = np.load('trained_model/scaling_params.npz')
bilstm = torch.load('trained_model/bilstm_production.pt', map_location='cpu', weights_only=False)
bilstm.eval()
classifier = torch.load('trained_model/classifier_production.pt', map_location='cpu', weights_only=False)
classifier.eval()

raw = np.array([4.5, 7.2, 26.5, 15.0, 0.5], dtype=np.float32)
scaled = (raw - scale_data['means']) / scale_data['stds']
seq = np.tile(scaled, (24,1))
tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

t0 = time.time()
with torch.no_grad():
    pred_do_scaled = bilstm(tensor)
    pred_do_actual = pred_do_scaled.item() * float(scale_data['do_std']) + float(scale_data['do_mean'])
    current_wq = tensor[:,-1,:]
    ids, probs, labels = classifier.predict(current_wq, pred_do_scaled)
ms = (time.time()-t0)*1000

print(f'SUCCESS!')
print(f'Predicted DO: {pred_do_actual:.4f} mg/L')
print(f'Risk: {labels[0]}')
print(f'Confidence: {probs[0, ids[0]].item()*100:.1f}%')
print(f'Latency: {ms:.1f}ms')
