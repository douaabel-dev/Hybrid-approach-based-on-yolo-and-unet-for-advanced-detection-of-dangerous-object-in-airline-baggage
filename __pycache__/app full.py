"""
xray_defence_web_v3.py — Thesis Defence Web Interface (Full Pipeline Edition)
==============================================================================
Detection interface + Data Preparation + Training Details + Arch Diagrams + Results Table

FIXES applied vs previous version:
  1. Modal onclick= attributes stripped by Gradio's HTML sanitizer →
     replaced with data-modal-open / data-modal-close attributes and
     a single event-delegation listener in the js= block.
  2. xray_20753.png invisible via /file= →
     image is base64-encoded at startup and injected as a data: URI.
"""

import os, cv2, warnings, base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.ops import box_iou, nms
from ultralytics import YOLO as _YOLO
from scipy import ndimage
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.linear_model import LogisticRegression
import gradio as gr
import PIL.Image as PILImage

warnings.filterwarnings("ignore")

YOLO_WEIGHTS      = "best.pt"
UNET_FULL_WEIGHTS = "best_unet_fullimage.pth"

CLASS_NAMES = ['Baton','Pliers','Hammer','Powerbank','Scissors',
               'Wrench','Gun','Bullet','Sprayer','HandCuffs','Knife','Lighter']
N_CLASSES   = len(CLASS_NAMES)

CONF_MAP   = 0.001
CONF_OPER  = 0.35
NMS_THRESH = 0.7
IMG_FULL   = 512
IMG_CROP   = 256
SEG_THRESH = 0.50
EPS        = 1e-7

LEARNED_CROP_PAD           = 12
LEARNED_CROP_SIZE          = 128
LEARNED_QUALITY_REJECT     = 0.25
LEARNED_CONF_HIGH_FALLBACK = 0.70
LEARNED_NMS_THRESH         = 0.35

SUPERVISOR_CONF_THRESH      = 0.25
SUPERVISOR_IOU_THRESH       = 0.45
SUPERVISOR_UNET_MASK_THRESH = 0.5
SUPERVISOR_OVERLAP_THRESH   = 0.15
SUPERVISOR_IMG_SIZE         = 640

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[app] Device: {device}")

tf_full = A.Compose([A.Resize(IMG_FULL, IMG_FULL), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
tf_crop = A.Compose([A.Resize(IMG_CROP, IMG_CROP), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
tf_learned_crop = A.Compose([A.Resize(LEARNED_CROP_SIZE, LEARNED_CROP_SIZE), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
import torchvision.transforms as tvtf
tf_supervisor = tvtf.Compose([tvtf.ToTensor(), tvtf.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ── Base64-encode the sample mask image at startup ────────────────────────────
# This avoids all /file= serving issues and works regardless of Gradio version.
_MASK_IMG_PATH = "xray_20753.png"
_MASK_IMG_B64  = ""
if os.path.exists(_MASK_IMG_PATH):
    with open(_MASK_IMG_PATH, "rb") as _f:
        _MASK_IMG_B64 = base64.b64encode(_f.read()).decode()
    print(f"[app] Mask sample image loaded ({len(_MASK_IMG_B64)//1024} KB base64)")
else:
    print(f"[app] WARNING: {_MASK_IMG_PATH} not found — placeholder will be shown")

def _mask_img_html() -> str:
    """Return an <img> tag (base64 inline) or a styled placeholder."""
    if _MASK_IMG_B64:
        return (
            f'<img src="data:image/png;base64,{_MASK_IMG_B64}" '
            f'style="max-width:100%;max-height:160px;border-radius:4px;object-fit:contain;" />'
        )
    # Fallback SVG placeholder — always visible
    return """
    <svg width="100%" height="140" viewBox="0 0 320 140" xmlns="http://www.w3.org/2000/svg"
         style="border-radius:6px;background:#050810;">
      <rect width="320" height="140" fill="#050810"/>
      <!-- simulate a binary mask blob -->
      <ellipse cx="160" cy="70" rx="90" ry="45" fill="#ffffff" opacity="0.88"/>
      <ellipse cx="110" cy="85" rx="30" ry="20" fill="#ffffff" opacity="0.75"/>
      <ellipse cx="220" cy="55" rx="25" ry="18" fill="#ffffff" opacity="0.70"/>
      <text x="160" y="128" text-anchor="middle"
            font-family="JetBrains Mono,monospace" font-size="10"
            fill="#2a4060">xray_20753.png — place file alongside script to load</text>
    </svg>"""

# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),nn.Conv2d(out_ch,out_ch,3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class UNetCustom(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs,self.ups = nn.ModuleList(),nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        ch = in_channels
        for f in features: self.downs.append(DoubleConv(ch,f)); ch=f
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2,f,2,2))
            self.ups.append(DoubleConv(f*2,f))
        self.final = nn.Conv2d(features[0],out_channels,1)
    def forward(self,x):
        skips=[]
        for down in self.downs: x=down(x); skips.append(x); x=self.pool(x)
        x=self.bottleneck(x); skips=skips[::-1]
        for i in range(0,len(self.ups),2):
            x=self.ups[i](x); s=skips[i//2]
            if x.shape!=s.shape: x=F.interpolate(x,size=s.shape[2:])
            x=self.ups[i+1](torch.cat([s,x],dim=1))
        return self.final(x)

def load_yolo(path=YOLO_WEIGHTS):
    m=_YOLO(path); m.to(device); print(f"[YOLO] {path}"); return m

def load_unet(path=UNET_FULL_WEIGHTS,label="UNet-SMP"):
    m=smp.Unet(encoder_name="resnet34",encoder_weights=None,in_channels=3,classes=1,activation=None).to(device)
    state=torch.load(path,map_location=device)
    if isinstance(state,dict) and 'model' in state: state=state['model']
    m.load_state_dict(state); m.eval(); print(f"[{label}] {path}"); return m

def load_unet_custom(path=UNET_FULL_WEIGHTS,label="UNet-Custom"):
    m=UNetCustom(in_channels=3,out_channels=1).to(device)
    ckpt=torch.load(path,map_location=device)
    if isinstance(ckpt,dict):
        for key in ("model_state_dict","model","state_dict","net"):
            if key in ckpt: state=ckpt[key]; break
        else: state=ckpt
    else: state=ckpt
    first_key=next(iter(state))
    if first_key.startswith("model."): state={k[6:]:v for k,v in state.items()}
    elif first_key.startswith("module."): state={k[7:]:v for k,v in state.items()}
    missing,_=m.load_state_dict(state,strict=False)
    print(f"[{label}] {len(state)-len(missing)}/{len(state)} keys loaded"); m.eval(); return m

print("Loading models …")
yolo_model      = load_yolo()
unet_model      = load_unet()
unet_cust_model = load_unet_custom()

def run_yolo(model,img_bgr,conf=CONF_MAP):
    r=model.predict(img_bgr,conf=conf,iou=NMS_THRESH,verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return (torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long))
    return (r.boxes.xyxy.cpu().float(),r.boxes.conf.cpu().float(),r.boxes.cls.cpu().long())

def run_unet_full(model,img_bgr):
    H,W=img_bgr.shape[:2]; rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    t=tf_full(image=rgb)['image'].unsqueeze(0).float().to(device)
    with torch.no_grad(): p=torch.sigmoid(model(t))[0,0].cpu()
    return F.interpolate(p.unsqueeze(0).unsqueeze(0),size=(H,W),mode='bilinear',align_corners=False)[0,0].numpy()

def run_unet_crop(model,img_bgr,box,pad=10):
    H_img,W_img=img_bgr.shape[:2]; x1,y1,x2,y2=[int(v) for v in box]
    cx1=max(0,x1-pad); cy1=max(0,y1-pad); cx2=min(W_img,x2+pad); cy2=min(H_img,y2+pad)
    if cx2<=cx1 or cy2<=cy1: return 0.0
    crop=cv2.cvtColor(img_bgr[cy1:cy2,cx1:cx2],cv2.COLOR_BGR2RGB)
    t=tf_crop(image=crop)['image'].unsqueeze(0).float().to(device)
    with torch.no_grad(): prob=torch.sigmoid(model(t))[0,0].cpu().numpy()
    ch,cw=cy2-cy1,cx2-cx1; prob_orig=cv2.resize(prob,(cw,ch))
    bx1=max(0,x1-cx1); by1=max(0,y1-cy1); bx2=min(cw,x2-cx1); by2=min(ch,y2-cy1)
    if bx2<=bx1 or by2<=by1: return 0.0
    region=prob_orig[by1:by2,bx1:bx2]
    return float((region>SEG_THRESH).sum())/(region.size+EPS)

def extract_components(binary_mask,prob_full,min_area=300):
    labeled,n=ndimage.label(binary_mask); boxes,probs=[],[]
    for cid in range(1,n+1):
        comp=(labeled==cid)
        if comp.sum()<min_area: continue
        rows,cols=np.where(comp)
        boxes.append(torch.tensor([float(cols.min()),float(rows.min()),float(cols.max()),float(rows.max())]))
        probs.append(float(prob_full[rows,cols].mean()))
    return boxes,probs

def _infer_class_for_new_det(ub,yolo_boxes,yolo_classes,cls_hist):
    if len(yolo_boxes)>0:
        uc_x=(ub[0]+ub[2])/2; uc_y=(ub[1]+ub[3])/2
        yc_x=(yolo_boxes[:,0]+yolo_boxes[:,2])/2; yc_y=(yolo_boxes[:,1]+yolo_boxes[:,3])/2
        dists=((yc_x-uc_x)**2+(yc_y-uc_y)**2).sqrt()
        nearest_idx=dists.argmin().item()
        if dists[nearest_idx].item()<200: return int(yolo_classes[nearest_idx].item())
    if cls_hist: return max(cls_hist,key=cls_hist.get)
    return 0

def _learned_crop_and_run_unet(model,img_bgr,box_xyxy):
    H,W=img_bgr.shape[:2]; x1,y1,x2,y2=box_xyxy
    cx1=max(0,int(x1)-LEARNED_CROP_PAD); cy1=max(0,int(y1)-LEARNED_CROP_PAD)
    cx2=min(W,int(x2)+LEARNED_CROP_PAD); cy2=min(H,int(y2)+LEARNED_CROP_PAD)
    if cx2-cx1<4 or cy2-cy1<4: return None,None
    crop_rgb=cv2.cvtColor(img_bgr[cy1:cy2,cx1:cx2],cv2.COLOR_BGR2RGB)
    t=tf_learned_crop(image=crop_rgb)['image'].unsqueeze(0).float().to(device)
    with torch.no_grad(): prob=torch.sigmoid(model(t))[0,0].cpu().numpy()
    return prob,(prob>SEG_THRESH).astype(np.uint8)

def _learned_compute_quality_features(prob,binary):
    fg_pixels=int(binary.sum()); total=binary.size; fg_ratio=fg_pixels/(total+EPS)
    if fg_ratio<LEARNED_QUALITY_REJECT: return None
    if fg_pixels>0:
        labeled,n=ndimage.label(binary)
        sizes=[int((labeled==i).sum()) for i in range(1,n+1)] if n>0 else [0]
        blob_size=max(sizes)/(fg_pixels+EPS)
    else: blob_size=0.0
    return float(fg_ratio),float(blob_size),float(prob.mean())

class LogisticFusion:
    def __init__(self):
        self.model=LogisticRegression(max_iter=1000,C=1.0); self.fitted=False; self.conf_high=LEARNED_CONF_HIGH_FALLBACK
    def score(self,yolo_conf,fg_ratio,blob_size,mask_mean):
        if not self.fitted:
            quality=0.40*fg_ratio+0.35*blob_size+0.25*mask_mean
            return 0.75*yolo_conf+0.25*quality
        return float(self.model.predict_proba(np.array([[yolo_conf,fg_ratio,blob_size,mask_mean]]))[0,1])

_learned_fusion_layer=LogisticFusion()

def fuse_yolo_only(img_bgr):
    yb,ys,yc=run_yolo(yolo_model,img_bgr); return yb,ys,yc,None

def fuse_arch1_score_refinement(img_bgr):
    HIGH_CONF=0.50; CONF_BOOST=0.06; CONF_PENALTY=0.18
    yb,ys,yc=run_yolo(yolo_model,img_bgr); fb,fs,fc=[],[],[]
    for i in range(len(yb)):
        box=yb[i]; score=ys[i].item(); cls=yc[i]
        if score>=HIGH_CONF: fb.append(box); fs.append(torch.tensor(score)); fc.append(cls); continue
        ratio=run_unet_crop(unet_model,img_bgr,box.tolist())
        if ratio>=0.30: score=min(1.,score+CONF_BOOST*ratio)
        else: score=max(0.,score-CONF_PENALTY*(1-ratio))
        fb.append(box); fs.append(torch.tensor(score)); fc.append(cls)
    if not fb: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),None
    return torch.stack(fb),torch.stack(fs),torch.stack(fc),None

def fuse_arch2_sequential_filtering(img_bgr):
    def _run_unet_sup(img_bgr_in):
        img_rgb=cv2.cvtColor(img_bgr_in,cv2.COLOR_BGR2RGB)
        img_rs=cv2.resize(img_rgb,(SUPERVISOR_IMG_SIZE,SUPERVISOR_IMG_SIZE))
        img_t=tf_supervisor(PILImage.fromarray(img_rs))
        with torch.no_grad():
            prob=torch.sigmoid(unet_cust_model(img_t.unsqueeze(0).to(device)))[0,0].cpu().numpy()
        return (prob>SUPERVISOR_UNET_MASK_THRESH).astype(np.uint8)
    H,W=img_bgr.shape[:2]
    r=yolo_model.predict(img_bgr,conf=SUPERVISOR_CONF_THRESH,iou=SUPERVISOR_IOU_THRESH,verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),None
    yb=r.boxes.xyxy.cpu().float(); ys=r.boxes.conf.cpu().float(); yc=r.boxes.cls.cpu().long()
    unet_mask=_run_unet_sup(img_bgr)
    pred_mask=cv2.resize(unet_mask.astype(np.float32),(W,H),interpolation=cv2.INTER_NEAREST)
    mH,mW=unet_mask.shape; ob,os_,oc=[],[],[]
    for pb,ps,pc in zip(yb,ys,yc):
        x1,y1,x2,y2=pb.tolist()
        mx1,my1,mx2,my2=int(x1/W*mW),int(y1/H*mH),int(x2/W*mW),int(y2/H*mH)
        mx1,my1=max(0,mx1),max(0,my1); mx2,my2=min(mW,mx2),min(mH,my2)
        if mx2<=mx1 or my2<=my1: continue
        roi=unet_mask[my1:my2,mx1:mx2]
        if roi.sum()/max(roi.size,1)>=SUPERVISOR_OVERLAP_THRESH: ob.append(pb); os_.append(ps); oc.append(pc)
    if not ob: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),pred_mask
    return torch.stack(ob),torch.stack(os_),torch.stack(oc),pred_mask

def fuse_arch3_parallel_score_refinement(img_bgr):
    OVERLAP_CONFIRM=0.20; OVERLAP_PEN=0.5; BOOST=0.15; PEN=0.01
    H,W=img_bgr.shape[:2]; yb,ys,yc=run_yolo(yolo_model,img_bgr)
    prob=run_unet_full(unet_model,img_bgr); pred_mask=(prob>SEG_THRESH).astype(np.float32)
    fb,fs,fc=[],[],[]
    for i in range(len(yb)):
        box=yb[i]; score=ys[i].item(); cls=yc[i]
        x1,y1,x2,y2=[int(v) for v in box.tolist()]
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        if x2>x1 and y2>y1:
            ov=float((prob[y1:y2,x1:x2]>SEG_THRESH).mean())
            if ov>=OVERLAP_CONFIRM: score=min(1.,score+BOOST*ov)
            elif ov<OVERLAP_PEN: score=max(0.,score-PEN*(1-ov))
        fb.append(box); fs.append(torch.tensor(score)); fc.append(cls)
    if not fb: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),pred_mask
    return torch.stack(fb),torch.stack(fs),torch.stack(fc),pred_mask

def fuse_arch4_learned_confidence(img_bgr):
    conf_high=_learned_fusion_layer.conf_high
    r=yolo_model.predict(img_bgr,conf=CONF_MAP,iou=LEARNED_NMS_THRESH,verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),None
    yb=r.boxes.xyxy.cpu().float(); ys=r.boxes.conf.cpu().float(); yc=r.boxes.cls.cpu().long()
    ob,os_,oc=[],[],[]
    for i in range(len(yb)):
        box=yb[i]; conf=float(ys[i]); cls=yc[i]
        if conf>=conf_high: ob.append(box); os_.append(conf); oc.append(cls); continue
        prob,binary=_learned_crop_and_run_unet(unet_model,img_bgr,box.tolist())
        if prob is None: ob.append(box); os_.append(conf); oc.append(cls); continue
        feats=_learned_compute_quality_features(prob,binary)
        if feats is None: continue
        fg_ratio,blob_size,mask_mean=feats
        S_final=_learned_fusion_layer.score(conf,fg_ratio,blob_size,mask_mean)
        if S_final<CONF_OPER: continue
        ob.append(box); os_.append(float(S_final)); oc.append(cls)
    if not ob: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),None
    boxes_t=torch.stack(ob); scores_t=torch.tensor(os_,dtype=torch.float32); classes_t=torch.stack(oc)
    keep_idx=[]
    for c in classes_t.unique():
        mask=classes_t==c; idx=torch.where(mask)[0]
        kept=nms(boxes_t[idx],scores_t[idx],LEARNED_NMS_THRESH); keep_idx.extend(idx[kept].tolist())
    keep_idx=sorted(keep_idx)
    return boxes_t[keep_idx],scores_t[keep_idx],classes_t[keep_idx],None

def fuse_arch5_attention_injection(img_bgr):
    ALPHA_K=8.0; ALPHA_PIVOT=0.50; ALPHA_MIN=0.55; SCORE_FLOOR=0.88
    IOU_MATCH=0.30; W_PROB=0.50; W_IOU=0.50
    NEW_CONF_MIN=0.82; MAX_NEW=3; MIN_COMP=400; IOU_EXISTING=0.20; MIN_CENTRE_DIST=150
    def _sig(x): return max(ALPHA_MIN,1./(1.+np.exp(-x)))
    H,W=img_bgr.shape[:2]; yb,ys,yc=run_yolo(yolo_model,img_bgr)
    prob=run_unet_full(unet_model,img_bgr); binary=(prob>SEG_THRESH).astype(np.uint8)
    unet_boxes,unet_probs=extract_components(binary,prob,MIN_COMP); pred_mask=binary.astype(np.float32)
    cls_hist={}
    for c in yc.tolist(): cls_hist[int(c)]=cls_hist.get(int(c),0)+1
    yolo_centres=[]
    if len(yb)>0: yolo_centres=torch.stack([(yb[:,0]+yb[:,2])/2,(yb[:,1]+yb[:,3])/2],dim=1)
    s1b_list,s1s_list,s1c_list=[],[],[]; used=set()
    for i in range(len(yb)):
        box=yb[i]; score=ys[i].item(); cls=yc[i]
        x1,y1,x2,y2=[int(v) for v in box.tolist()]
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        mean_prob=float(prob[y1:y2,x1:x2].mean()) if x2>x1 and y2>y1 else 0.
        alpha=_sig(ALPHA_K*(score-ALPHA_PIVOT)); iou_support=0.
        if unet_boxes:
            ut=torch.stack(unet_boxes); ious=box_iou(box.unsqueeze(0),ut)[0]
            bv,bj=ious.max().item(),ious.argmax().item()
            if bv>=IOU_MATCH: iou_support=bv; used.add(bj)
        unet_signal=W_PROB*mean_prob+W_IOU*iou_support
        blended=float(np.clip(alpha*score+(1-alpha)*unet_signal,0.,1.))
        blended=max(blended,score*SCORE_FLOOR)
        if score>=CONF_OPER and blended<CONF_OPER: blended=max(CONF_OPER+0.01,score*0.95)
        s1b_list.append(box); s1s_list.append(blended); s1c_list.append(cls)
    if s1b_list:
        s1b=torch.stack(s1b_list); s1s=torch.tensor(s1s_list); s1c=torch.stack(s1c_list); keep_idx=[]
        for c in s1c.unique():
            mask=s1c==c; idx=torch.where(mask)[0]
            kept=nms(s1b[idx],s1s[idx],NMS_THRESH); keep_idx.extend(idx[kept].tolist())
        keep_idx=sorted(keep_idx); s1b=s1b[keep_idx]; s1s=s1s[keep_idx]; s1c=s1c[keep_idx]
    else: s1b=torch.zeros((0,4)); s1s=torch.zeros(0); s1c=torch.zeros(0,dtype=torch.long)
    new_boxes,new_scores,new_classes=[],[],[]; new_count=0
    for j,(ub,cp) in enumerate(zip(unet_boxes,unet_probs)):
        if j in used or cp<NEW_CONF_MIN or new_count>=MAX_NEW: continue
        if len(yolo_centres)>0:
            ub_cx=(ub[0]+ub[2])/2; ub_cy=(ub[1]+ub[3])/2
            dists=((yolo_centres[:,0]-ub_cx)**2+(yolo_centres[:,1]-ub_cy)**2).sqrt()
            if dists.min().item()<MIN_CENTRE_DIST: continue
        if len(s1b)>0 and box_iou(ub.unsqueeze(0),s1b)[0].max().item()>IOU_EXISTING: continue
        assigned_cls=_infer_class_for_new_det(ub,yb,yc,cls_hist)
        if assigned_cls not in cls_hist: continue
        new_boxes.append(ub); new_scores.append(float(np.clip(cp*0.75,0.,1.)))
        new_classes.append(torch.tensor(assigned_cls,dtype=torch.long)); new_count+=1
    all_boxes=list(s1b)+new_boxes; all_scores=list(s1s.numpy())+new_scores; all_cls=list(s1c)+new_classes
    if not all_boxes: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),pred_mask
    return torch.stack(all_boxes),torch.tensor(all_scores,dtype=torch.float32),torch.stack(all_cls),pred_mask

def fuse_arch6_controlled_recall(img_bgr):
    ALPHA_K=6.0; ALPHA_PIVOT=0.40; ALPHA_MIN=0.55; SCORE_FLOOR=0.92
    IOU_MATCH=0.25; W_PROB=0.40; W_IOU=0.60; MIN_COMP=200
    RESCUE_IOU=0.30; RESCUE_CONF=0.60; RESCUE_BOOST=0.18
    def _sig(x): return max(ALPHA_MIN,1./(1.+np.exp(-x)))
    H,W=img_bgr.shape[:2]; yb,ys,yc=run_yolo(yolo_model,img_bgr)
    prob=run_unet_full(unet_model,img_bgr); binary=(prob>SEG_THRESH).astype(np.uint8)
    unet_boxes,_=extract_components(binary,prob,MIN_COMP); pred_mask=binary.astype(np.float32)
    if len(yb)==0: return torch.zeros((0,4)),torch.zeros(0),torch.zeros(0,dtype=torch.long),pred_mask
    bb,bs,bc=[],[],[]
    for i in range(len(yb)):
        box=yb[i]; score=ys[i].item(); cls=yc[i]
        x1,y1,x2,y2=[int(v) for v in box.tolist()]
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        mean_prob=float(prob[y1:y2,x1:x2].mean()) if x2>x1 and y2>y1 else 0.
        alpha=_sig(ALPHA_K*(score-ALPHA_PIVOT)); iou_support=0.
        if unet_boxes:
            ut=torch.stack(unet_boxes); ious=box_iou(box.unsqueeze(0),ut)[0]
            best=ious.max().item()
            if best>=IOU_MATCH: iou_support=best
        unet_signal=W_PROB*mean_prob+W_IOU*iou_support
        if iou_support>RESCUE_IOU and score<RESCUE_CONF: score=min(1.0,score+RESCUE_BOOST*iou_support)
        blended=float(np.clip(alpha*score+(1-alpha)*unet_signal,0.,1.))
        blended=max(blended,score*SCORE_FLOOR)
        if score>=CONF_OPER and blended<CONF_OPER: blended=max(CONF_OPER+0.02,score*0.95)
        bb.append(box); bs.append(blended); bc.append(cls)
    b_t=torch.stack(bb); s_t=torch.tensor(bs,dtype=torch.float32); c_t=torch.stack(bc); keep_idx=[]
    for c in c_t.unique():
        mask=c_t==c; idx=torch.where(mask)[0]
        kept=nms(b_t[idx],s_t[idx],NMS_THRESH); keep_idx.extend(idx[kept].tolist())
    keep_idx=sorted(keep_idx)
    return b_t[keep_idx],s_t[keep_idx],c_t[keep_idx],pred_mask

ARCHS = {
    "YOLO Baseline":(fuse_yolo_only,"Pure YOLO detection. Reference baseline — no segmentation fusion.",False,[("mAP@0.5","0.8023"),("Precision","0.8113"),("Recall","0.7887"),("F1","0.7998")]),
    "Arch. 1: Score Refinement":(fuse_arch1_score_refinement,"Precision-Oriented Family. Crop U-Net adjusts YOLO confidence up/down based on foreground overlap ratio within each detected bounding box.",False,[("mAP@0.5","0.7843"),("Precision","0.8539"),("Recall","0.7655"),("F1","0.8073")]),
    "Arch. 2: Sequential Filtering":(fuse_arch2_sequential_filtering,"Precision-Oriented Family. Custom U-Net mask gates YOLO boxes: a detection is kept only if its region overlaps the segmentation mask by ≥15%. Highest F1.",True,[("mAP@0.5","0.7475"),("Precision","0.8376"),("Recall","0.7848"),("F1","0.8104 ★")]),
    "Arch. 3: Parallel Score Refinement":(fuse_arch3_parallel_score_refinement,"Precision-Oriented Family. Full-image U-Net runs in parallel with YOLO. Scores boosted or penalised by foreground pixel overlap.",True,[("mAP@0.5","0.7932"),("Precision","0.8315"),("Recall","0.7805"),("F1","0.8052")]),
    "Arch. 4: Learned Confidence Fusion":(fuse_arch4_learned_confidence,"Precision-Oriented Family. Logistic regression calibrates (YOLO conf, fg_ratio, blob_size, mask_mean) → TP probability. Highest precision.",False,[("mAP@0.5","0.6605"),("Precision","0.9263 ★"),("Recall","0.6858"),("F1","0.7881")]),
    "Arch. 5: Attention Fusion + Injection":(fuse_arch5_attention_injection,"Recall-Oriented Family. Adaptive sigmoid gate blends YOLO and U-Net scores (ALPHA_MIN=0.55). Injects new detections from unmatched U-Net blobs. Highest recall.",True,[("mAP@0.5","0.7944"),("Precision","0.4566"),("Recall","0.8448 ★"),("F1","0.5928")]),
    "Arch. 6: Controlled Recall Recovery":(fuse_arch6_controlled_recall,"Recall-Oriented Family. Attention fusion with rescue mechanism: boosts score of YOLO detections with strong U-Net IoU support but low initial confidence.",True,[("mAP@0.5","0.7963 ★"),("Precision","0.6195"),("Recall","0.8262"),("F1","0.7081")]),
}
ARCH_NAMES=list(ARCHS.keys())

COLOURS=[(0,200,255),(0,255,128),(255,128,0),(128,0,255),(0,255,255),(255,0,128),(0,128,255),(255,255,0),(128,255,0),(0,0,255),(255,0,0),(0,255,0)]

def _draw_boxes(img_bgr,boxes,scores,classes,title="",mask=None):
    vis=img_bgr.copy()
    if mask is not None:
        hot=(mask>SEG_THRESH).astype(np.uint8)
        vis[hot==1]=(vis[hot==1]*0.55+np.array([0,180,0])*0.45).astype(np.uint8)
    n_det=0
    for box,score,cls in zip(boxes,scores,classes):
        if float(score)<CONF_OPER: continue
        n_det+=1; x1,y1,x2,y2=[int(v) for v in box.tolist()]
        cls_id=int(cls); col=COLOURS[cls_id%len(COLOURS)]
        nm=CLASS_NAMES[cls_id] if cls_id<N_CLASSES else '?'
        cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
        label=f"{nm} {float(score):.2f}"
        (lw,lh),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
        cv2.rectangle(vis,(x1,y1-lh-6),(x1+lw+4,y1),col,-1)
        cv2.putText(vis,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
    bar_txt=f"{title}  [{n_det} detection{'s' if n_det!=1 else ''}]"
    (bw,bh),_=cv2.getTextSize(bar_txt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    cv2.rectangle(vis,(0,0),(vis.shape[1],bh+12),(15,15,25),-1)
    cv2.putText(vis,bar_txt,(8,bh+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,220,255),1)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def _draw_mask(img_bgr,prob):
    vis=img_bgr.copy(); mask=(prob>SEG_THRESH).astype(np.uint8)
    fg=mask.sum(); total=mask.size; cover=fg/total*100
    vis[mask==1]=(vis[mask==1]*0.35+np.array([0,210,60])*0.65).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis,contours,-1,(0,255,100),2)
    title=f"U-Net Segmentation Mask  [{cover:.1f}% foreground]"
    (bw,bh),_=cv2.getTextSize(title,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    cv2.rectangle(vis,(0,0),(vis.shape[1],bh+12),(15,15,25),-1)
    cv2.putText(vis,title,(8,bh+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,150),1)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def predict(pil_image,arch_name):
    if pil_image is None:
        blank=np.zeros((480,640,3),dtype=np.uint8)
        cv2.putText(blank,"No image loaded",(180,240),cv2.FONT_HERSHEY_SIMPLEX,1.0,(80,80,80),2)
        return blank,blank,blank,"⚠️ Upload an X-ray image to begin."
    img_rgb=np.array(pil_image); img_bgr=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
    fusion_fn,desc,has_mask,metrics=ARCHS[arch_name]
    yb,ys,yc=run_yolo(yolo_model,img_bgr)
    n_yolo=int((ys>=CONF_OPER).sum())
    yolo_vis=_draw_boxes(img_bgr,yb,ys,yc,title="YOLO Baseline")
    fb,fs,fc,pred_mask=fusion_fn(img_bgr)
    n_hybrid=int((fs>=CONF_OPER).sum()); delta=n_hybrid-n_yolo
    delta_str=f"+{delta}" if delta>=0 else str(delta)
    arch_short=arch_name.split(":")[0].strip() if ":" in arch_name else arch_name
    hybrid_vis=_draw_boxes(img_bgr,fb,fs,fc,title=f"{arch_short}  (Δ{delta_str})",mask=pred_mask)
    if has_mask and pred_mask is not None: mask_vis=_draw_mask(img_bgr,pred_mask)
    elif arch_name!=ARCH_NAMES[0]:
        prob_for_display=run_unet_full(unet_model,img_bgr); mask_vis=_draw_mask(img_bgr,prob_for_display)
    else:
        mask_vis=img_bgr.copy()
        cv2.rectangle(mask_vis,(0,0),(mask_vis.shape[1],36),(15,15,25),-1)
        cv2.putText(mask_vis,"U-Net mask not used by baseline",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,120),1)
        mask_vis=cv2.cvtColor(mask_vis,cv2.COLOR_BGR2RGB)
    metrics_rows="\n".join(f"| {k} | **{v}** |" for k,v in metrics)
    summary=f"""## {arch_name}\n\n{desc}\n\n| Metric | Value |\n|:--|:--|\n{metrics_rows}\n\n---\n\n**YOLO baseline:** {n_yolo} detection{'s' if n_yolo!=1 else ''}  \n**{arch_short}:** {n_hybrid} detection{'s' if n_hybrid!=1 else ''}  \n**Delta vs baseline:** {delta_str} object{'s' if abs(delta)!=1 else ''}\n\n*Confidence threshold: {CONF_OPER} · NMS IoU: {NMS_THRESH}*"""
    return yolo_vis,mask_vis,hybrid_vis,summary

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════
CSS = """
:root{--bg-primary:#070b14;--bg-card:#0d1424;--bg-card-hover:#111a30;--border:#1e2d4a;--accent:#00c8ff;--accent-2:#00ff9d;--accent-warn:#ff6b35;--text-primary:#e8eef8;--text-muted:#7a8ba8;--font-mono:'JetBrains Mono','Fira Code',monospace;}
body,.gradio-container{background:var(--bg-primary)!important;font-family:'IBM Plex Sans','Segoe UI',sans-serif;}
footer{display:none!important;}
.header-block{text-align:center;padding:2rem 1rem 1rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;}
.header-block h1{font-size:2rem;font-weight:700;letter-spacing:-0.5px;color:var(--text-primary);margin:0;}
.header-block .accent{color:var(--accent);}
.badge{display:inline-block;background:rgba(0,200,255,0.12);border:1px solid rgba(0,200,255,0.3);color:var(--accent);border-radius:20px;padding:2px 12px;font-size:0.75rem;font-family:var(--font-mono);margin:0.5rem 0.2rem 0;}
.panel{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:14px!important;padding:1rem!important;}
.output-image img{border-radius:10px;border:1px solid var(--border);}
.arch-radio .wrap{gap:6px!important;}
.arch-radio label{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:8px 12px!important;font-size:0.82rem!important;color:var(--text-primary)!important;transition:all 0.2s ease;cursor:pointer;}
.arch-radio label:hover{border-color:var(--accent)!important;background:var(--bg-card-hover)!important;}
.arch-radio input:checked+label,.arch-radio label.selected{border-color:var(--accent)!important;color:var(--accent)!important;background:rgba(0,200,255,0.07)!important;}
.run-btn{background:linear-gradient(135deg,#0070f3,#00c8ff)!important;border:none!important;border-radius:10px!important;color:white!important;font-weight:700!important;font-size:1rem!important;padding:12px 0!important;letter-spacing:0.3px;transition:opacity 0.2s;}
.run-btn:hover{opacity:0.88;}
.upload-box{border:1px dashed var(--border)!important;border-radius:12px!important;background:var(--bg-card)!important;}
.metric-md,.metric-md>*,.metric-md .prose,.metric-md .prose>*{background:var(--bg-card)!important;border-color:var(--border)!important;color:var(--text-primary)!important;}
.metric-md{border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem!important;}
.metric-md p,.metric-md span,.metric-md li,.metric-md em,.metric-md div{color:var(--text-primary)!important;}
.metric-md h1,.metric-md h2,.metric-md h3{color:var(--accent)!important;font-size:1rem!important;font-weight:700!important;margin-bottom:0.5rem!important;border-bottom:1px solid var(--border)!important;padding-bottom:4px!important;}
.metric-md strong{color:#ffffff!important;font-weight:700!important;}
.metric-md hr{border-color:var(--border)!important;margin:0.6rem 0!important;}
.metric-md table{width:100%!important;border-collapse:collapse!important;font-family:var(--font-mono)!important;font-size:0.85rem!important;background:transparent!important;}
.metric-md th,.metric-md td{border:1px solid var(--border)!important;padding:6px 12px!important;text-align:left!important;color:var(--text-primary)!important;background:transparent!important;}
.metric-md th{background:rgba(0,200,255,0.08)!important;color:var(--accent)!important;font-weight:600!important;}
.metric-md td strong{color:#00ff9d!important;}
.legend{background:rgba(0,200,255,0.04);border:1px solid rgba(0,200,255,0.15);border-radius:10px;padding:10px 14px;font-size:0.8rem;color:var(--text-muted);margin-top:12px;}
.arch-table table{width:100%;border-collapse:collapse;font-size:0.83rem;font-family:var(--font-mono);}
.arch-table th{background:rgba(0,200,255,0.08);color:var(--accent);border-bottom:1px solid var(--border);padding:8px 14px;text-align:left;}
.arch-table td{border-bottom:1px solid rgba(30,45,74,0.5);padding:7px 14px;color:var(--text-primary);}
.arch-table tr:hover td{background:var(--bg-card-hover);}
.panel-label{font-family:var(--font-mono);font-size:0.75rem;letter-spacing:1.5px;text-transform:uppercase;color:var(--text-muted);margin-bottom:6px;}
.panel-label.yolo{color:#ff9500;}.panel-label.mask{color:var(--accent-2);}.panel-label.hybrid{color:var(--accent);}
.section-wrap{margin-top:2.8rem;}
.section-title{font-family:var(--font-mono);font-size:0.68rem;letter-spacing:3px;text-transform:uppercase;color:var(--text-muted);margin-bottom:1rem;padding-bottom:8px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;}
.section-title::before{content:'';display:inline-block;width:3px;height:14px;background:var(--accent);border-radius:2px;}

/* ── pip-card: clickable with pointer cursor ── */
.pip-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.2rem;position:relative;transition:border-color 0.2s,transform 0.15s;cursor:pointer;box-sizing:border-box;}
.pip-card:hover{border-color:var(--accent);transform:translateY(-2px);}
.pip-card .card-icon{font-size:1.4rem;margin-bottom:0.5rem;}
.pip-card .card-title{font-size:0.78rem;font-weight:700;color:var(--accent);letter-spacing:0.5px;margin-bottom:0.4rem;text-transform:uppercase;font-family:var(--font-mono);}
.pip-card .card-body{font-size:0.82rem;color:var(--text-muted);line-height:1.55;}
.pip-card .tag{display:inline-block;background:rgba(0,200,255,0.1);border:1px solid rgba(0,200,255,0.25);color:var(--accent);border-radius:4px;padding:1px 7px;font-size:0.68rem;font-family:var(--font-mono);margin:2px 2px 0 0;}
.pip-card .tag.green{background:rgba(0,255,157,0.1);border-color:rgba(0,255,157,0.25);color:var(--accent-2);}
.pip-card .tag.orange{background:rgba(255,107,53,0.1);border-color:rgba(255,107,53,0.25);color:var(--accent-warn);}

/* ── Modal styles ── */
.modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(5,8,16,0.92);z-index:9999;align-items:center;justify-content:center;backdrop-filter:blur(4px);}
.modal-overlay.active{display:flex;}
.modal-box{background:#0a1020;border:1px solid var(--border);border-radius:16px;padding:1.8rem;max-width:860px;width:92%;max-height:86vh;overflow-y:auto;position:relative;}
.modal-box h3{color:var(--accent);font-family:var(--font-mono);font-size:0.85rem;letter-spacing:1px;text-transform:uppercase;margin:0 0 1rem 0;}
.modal-box pre{background:#050810;border:1px solid #1a2540;border-radius:8px;padding:1rem;font-size:0.73rem;color:#a8c8f0;overflow-x:auto;font-family:var(--font-mono);line-height:1.6;white-space:pre;}
.modal-close{position:absolute;top:1rem;right:1.2rem;background:none;border:1px solid var(--border);color:var(--text-muted);border-radius:6px;padding:3px 10px;cursor:pointer;font-size:0.8rem;transition:border-color 0.2s;}
.modal-close:hover{border-color:var(--accent);color:var(--accent);}
.modal-section-title{font-family:var(--font-mono);font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin:1.2rem 0 0.5rem;padding-bottom:4px;border-bottom:1px solid var(--border);}
.stat-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(30,45,74,0.4);font-size:0.82rem;}
.stat-row:last-child{border-bottom:none;}
.stat-label{color:var(--text-muted);}
.stat-val{color:var(--text-primary);font-family:var(--font-mono);font-weight:600;}
.stat-val.accent{color:var(--accent);}
.arch-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem 1.2rem;transition:border-color 0.2s,transform 0.15s;}
.arch-card:hover{border-color:var(--accent);transform:translateY(-2px);}
.arch-card-header{display:flex;align-items:center;gap:8px;margin-bottom:0.7rem;}
.arch-num{font-family:var(--font-mono);font-size:0.7rem;color:var(--accent);background:rgba(0,200,255,0.1);border:1px solid rgba(0,200,255,0.25);border-radius:4px;padding:2px 8px;}
.arch-card-title{font-size:0.8rem;font-weight:700;color:var(--text-primary);}
.arch-family{font-size:0.68rem;color:var(--text-muted);font-family:var(--font-mono);margin-bottom:0.55rem;}
.arch-family.precision{color:#f59e0b;}.arch-family.recall{color:#34d399;}
.flow{display:flex;align-items:center;gap:0;flex-wrap:nowrap;margin:0.6rem 0;overflow-x:auto;}
.flow-box{background:#0a1525;border:1px solid var(--border);border-radius:6px;padding:4px 7px;font-size:0.65rem;color:var(--text-muted);white-space:nowrap;flex-shrink:0;font-family:var(--font-mono);}
.flow-box.yolo{border-color:#f59e0b55;color:#f59e0b;background:rgba(245,158,11,0.07);}
.flow-box.unet{border-color:#34d39955;color:#34d399;background:rgba(52,211,153,0.07);}
.flow-box.fuse{border-color:#00c8ff55;color:#00c8ff;background:rgba(0,200,255,0.07);}
.flow-box.out{border-color:#a78bfa55;color:#a78bfa;background:rgba(167,139,250,0.07);}
.flow-arr{color:#2a3a55;font-size:0.75rem;padding:0 2px;flex-shrink:0;}
.arch-metrics-mini{display:flex;gap:5px;margin-top:0.65rem;flex-wrap:wrap;}
.mini-metric{background:#050810;border:1px solid var(--border);border-radius:5px;padding:3px 7px;font-size:0.66rem;font-family:var(--font-mono);}
.mini-metric .mk{color:var(--text-muted);}
.mini-metric .mv{color:var(--text-primary);font-weight:700;margin-left:3px;}
.mini-metric .mv.best{color:var(--accent-2);}
"""

# ═══════════════════════════════════════════════════════════════════════
# JS — injected via gr.Blocks(js=...)
#
# FIX: Gradio strips inline onclick= attributes from gr.HTML() blocks.
# Solution: use data-modal-open / data-modal-close attributes and a
# single event-delegation listener on document.  data-* attributes are
# NOT stripped by Gradio's sanitizer.
# ═══════════════════════════════════════════════════════════════════════
MODAL_JS = """
function() {
    /* ── Modal open/close helpers ── */
    window.openModal = function(id) {
        var el = document.getElementById(id);
        if (el) { el.classList.add('active'); document.body.style.overflow = 'hidden'; }
    };
    window.closeModal = function(id) {
        var el = document.getElementById(id);
        if (el) { el.classList.remove('active'); document.body.style.overflow = ''; }
    };

    /* ── Event delegation (works because data-* attrs survive Gradio sanitization) ── */
    document.addEventListener('click', function(e) {
        /* Open: click on any element with data-modal-open="modal-id" */
        var opener = e.target.closest('[data-modal-open]');
        if (opener) {
            window.openModal(opener.getAttribute('data-modal-open'));
            return;
        }
        /* Close: click on any element with data-modal-close="modal-id" */
        var closer = e.target.closest('[data-modal-close]');
        if (closer) {
            window.closeModal(closer.getAttribute('data-modal-close'));
            return;
        }
        /* Close: click on the overlay background itself */
        var overlay = e.target.closest('.modal-overlay');
        if (overlay && e.target === overlay) {
            window.closeModal(overlay.id);
            return;
        }
    });

    /* ── Escape key closes any open modal ── */
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal-overlay.active').forEach(function(m) {
                m.classList.remove('active');
            });
            document.body.style.overflow = '';
        }
    });
}
"""

# ═══════════════════════════════════════════════════════════════════════
# HTML BLOCKS
# NOTE: All onclick= replaced with data-modal-open= / data-modal-close=
# ═══════════════════════════════════════════════════════════════════════

HEADER_HTML = """
<div class="header-block">
  <h1> <span class="accent">Hybrid YOLO–U-Net Airport Security Detection System</span> </h1>
  <div style="margin-top:0.7rem">
    <span class="badge">YOLO v11n</span>
    <span class="badge">U-Net</span>
    <span class="badge">6 Architectures</span>
    <span class="badge">12 Threat Classes</span>
  </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════
# §1 — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════
DATA_PREP_HTML = f"""
<div class="section-wrap">
  <div class="section-title">§1 &mdash; Data Preparation</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">

    <!-- Card 1: Annotation JSON — data-modal-open replaces onclick -->
    <div class="pip-card" data-modal-open="modal-annot-json">
      <div class="card-title">PIDray Annotation Structure</div>
      <div class="card-body">
        <pre style="background:#050810;border:1px solid #1a2540;border-radius:6px;padding:0.6rem;font-size:0.62rem;color:#a8c8f0;font-family:var(--font-mono);line-height:1.5;overflow-x:auto;white-space:pre-wrap;word-break:break-all;margin:0 0 0.7rem;">{{"id":0,"image_id":0,"category_id":1,
"segmentation":[[202.8904109589041, 127.08219178082192, 176.86301369863014, 307.9041095890411, 170.6986301369863, 316.8082191780822, 160.4246575342466, 370.2328767123288, 198.78082191780823, 382.56164383561645, 207.68493150684932, 376.3972602739726, 217.27397260273972, 323.6575342465753, 214.53424657534248, 315.43835616438355, 241.24657534246575, 130.5068493150685]],
"bbox":[160.42,127.08,80.82,255.48],
"iscrowd":0,"area":10213.0}}</pre>
        <span class="tag">COCO JSON format</span>
        <span class="tag">12 classes</span>
        <br>
        <span class="tag">70 001 images</span>
        <span class="tag">39 708 annotations</span>
        <br><br>
        <strong style="font-size:0.74rem;color:var(--text-primary)">3 main keys:</strong>
        <br>
        <span style="font-family:var(--font-mono);font-size:0.72rem;color:#f59e0b">annotations</span> — id, image_id, category_id, segmentation polygon, bbox, area<br>
        <span style="font-family:var(--font-mono);font-size:0.72rem;color:#34d399">categories</span> — 12 threat class labels (id 1–12)<br>
        <span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--accent)">images</span> — height, width, id, file_name<br><br>
      </div>
    </div>

<!-- Card 2: YOLO Conversion -->
<div class="pip-card" data-modal-open="modal-yolo-annot">
  <div class="card-title">YOLO Label Formatting</div>
  <div class="card-body">
    COCO bounding boxes are converted into the YOLO normalized
    annotation format. A `.txt` label file is generated for each
    image, with one line per object instance.
    <br><br>

    <span class="tag green">Normalized center coordinates</span>
    <span class="tag green">YOLO bbox format</span>
    <span class="tag green">0-indexed class IDs</span>

    <br><br>

    <strong style="font-size:0.74rem;color:var(--text-primary)">
      YOLO Format:
    </strong><br>

    <span style="font-family:var(--font-mono);font-size:0.72rem;color:#a8d8ff">
      class_id x_center y_center width height
    </span>

    <br><br>

    <strong style="font-size:0.74rem;color:var(--text-primary)">
      Conversion:
    </strong><br>

    <span style="font-family:var(--font-mono);font-size:0.72rem;color:#a8d8ff">
      x_center = (x + w/2) / img_w<br>
      y_center = (y + h/2) / img_h<br>
      class_id = category_id - 1
    </span>
  </div>
</div>

     <!-- Card 3: Segmentation Masks -->
<div class="pip-card" data-modal-open="modal-seg-mask">
  <div class="card-title">Segmentation Mask Generation</div>

  <div class="card-body">
    Segmentation masks are created from COCO polygon annotations
    to train the U-Net model for pixel-level object localisation.
    Each annotated polygon is converted into a binary foreground mask
    using cv2.fillPoly(), where dangerous objects are
    represented by white pixels (255) and the
    background by black pixels (0).

    <br><br>

    <span class="tag orange">Pixel-level annotations</span>
    <span class="tag orange">Foreground = 255</span>
    <span class="tag orange">Background = 0</span>

    <br><br>

    <strong style="font-size:0.74rem;color:var(--text-primary)">
      Processing:
    </strong><br>

    <span style="font-size:0.73rem;color:var(--text-muted)">
      1. Initialize empty mask per image<br>
      2. Convert polygon coordinates to contours<br>
      3. Fill object regions with white pixels<br>
      4. Save binary PNG mask at original resolution
    </span>

    <br><br>
  </div>
</div>
  </div>

  <!-- Sample output row -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;">

    <!-- YOLO annotation sample (non-clickable info card) -->
    <div class="pip-card" style="cursor:default;">
      <div class="card-icon">📄</div>
      <div class="card-title">YOLO Annotation File Sample</div>
      <div class="card-body">
        Each line: class_id &nbsp; x_center &nbsp; y_center &nbsp; width &nbsp; height — all values normalised to [0, 1].
        <br><br>
        <pre style="background:#050810;border:1px solid #1a2540;border-radius:8px;padding:0.9rem;font-size:0.76rem;color:#a8d8ff;font-family:var(--font-mono);margin:0.4rem 0 0.6rem;line-height:2;overflow-x:auto;">
0   0.4416   0.4615   0.5255   0.1163
0   0.5380   0.3244   0.2878   0.2949
0   0.4521   0.5880   0.3255   0.1116
6   0.3125   0.2890   0.1430   0.3012
7   0.7161   0.6271   0.1538   0.3196
2   0.3935   0.5142   0.4175   0.6165
10  0.2701   0.4206   0.1253   0.4931</pre>
        <div style="display:flex;gap:14px;margin-top:6px;font-size:0.73rem;color:var(--text-muted);font-family:var(--font-mono);flex-wrap:wrap;">
          <span><span style="color:var(--accent)">col 0 &nbsp;class_id (0-indexed)</span></span>
          <span><span style="color:var(--accent-2)">col 1-2 &nbsp;x_center, y_center</span></span>
          <span><span style="color:#f59e0b">col 3-4 &nbsp;width, height</span></span>
        </div>
      </div>
    </div>

    <!-- Segmentation mask sample — image injected as base64 data: URI -->
    <div class="pip-card" style="cursor:default;">
      <div class="card-icon">🖼️</div>
      <div class="card-title">Segmentation Mask Sample</div>
      <div class="card-body">
        Binary PNG — <span style="color:#fff;font-weight:600">white (255)</span> = threat object foreground,
        <span style="color:#7a8ba8">black (0)</span> = background.
        <br><br>
        <div style="border:1px solid #1a2540;border-radius:8px;overflow:hidden;text-align:center;background:#050810;padding:10px;">
          {_mask_img_html()}
        </div>
        <div style="margin-top:8px;font-size:0.72rem;color:var(--text-muted);">
          <span style="display:inline-block;width:10px;height:10px;background:#fff;border-radius:2px;margin-right:4px;vertical-align:middle;"></span>White = foreground (threat object)&emsp;
          <span style="display:inline-block;width:10px;height:10px;background:#1a2540;border-radius:2px;margin-right:4px;vertical-align:middle;border:1px solid #2a3a55"></span>Black = background
        </div>
      </div>
    </div>
  </div>
</div>

"""

# ═══════════════════════════════════════════════════════════════════════
# §2 — TRAINING DETAILS
# ═══════════════════════════════════════════════════════════════════════
TRAINING_HTML = """
<div class="section-wrap">
  <div class="section-title">§2 &mdash; Training Details</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">

    <!-- YOLO Training Card -->
<div class="pip-card" data-modal-open="modal-yolo-train">
  <div class="card-title">YOLOv11n — Object Detection Training</div>

  <div class="card-body">
    The YOLO model is trained for multi-class object detection on X-ray baggage images using a COCO-to-YOLO converted dataset with 12 threat categories.

    <br><br>

    <span class="tag green">12 object classes</span>
    <span class="tag green">640×640 input</span>
    <span class="tag green">COCO → YOLO format</span>
    <span class="tag green">Real-time detection model</span>

    <br><br>

    <div class="stat-row"><span class="stat-label">Model</span><span class="stat-val accent">YOLOv11n (pretrained)</span></div>
    <div class="stat-row"><span class="stat-label">Dataset</span><span class="stat-val">PIDray (train/val split)</span></div>
    <div class="stat-row"><span class="stat-label">Input size</span><span class="stat-val">640 × 640</span></div>
    <div class="stat-row"><span class="stat-label">Epochs</span><span class="stat-val">100</span></div>
    <div class="stat-row"><span class="stat-label">Batch size</span><span class="stat-val">16</span></div>
    <div class="stat-row"><span class="stat-label">Optimizer</span><span class="stat-val">SGD (default)</span></div>

    <div class="stat-row"><span class="stat-label">Augmentations</span><span class="stat-val">
      Mosaic · MixUp · HSV shift · Flip · Scale · Shear
    </span></div>

    <div class="stat-row"><span class="stat-label">Loss</span><span class="stat-val">
      Box + Classification + DFL (Ultralytics)
    </span></div>

    <div class="stat-row"><span class="stat-label">Hardware</span><span class="stat-val">Kaggle T4 × 2 GPU</span></div>
  </div>
</div>

    <!-- U-Net Training Card -->
<div class="pip-card" data-modal-open="modal-unet-train">
  <div class="card-title">U-Net (ResNet34) — Segmentation Training</div>

  <div class="card-body">
    The U-Net model is trained for pixel-level segmentation to generate binary masks of dangerous objects from X-ray images.

    <br><br>

    <span class="tag blue">Binary segmentation</span>
    <span class="tag blue">Foreground vs background</span>
    <span class="tag blue">ResNet34 encoder</span>
    <span class="tag blue">Dice + BCE loss</span>

    <br><br>

    <div class="stat-row"><span class="stat-label">Architecture</span><span class="stat-val accent">U-Net + ResNet34 encoder</span></div>
    <div class="stat-row"><span class="stat-label">Dataset</span><span class="stat-val">COCO polygons → binary masks</span></div>
    <div class="stat-row"><span class="stat-label">Input size</span><span class="stat-val">512 × 512</span></div>
    <div class="stat-row"><span class="stat-label">Epochs</span><span class="stat-val">40 (early stopping)</span></div>
    <div class="stat-row"><span class="stat-label">Batch size</span><span class="stat-val">8 (effective 16)</span></div>

    <div class="stat-row"><span class="stat-label">Optimizer</span><span class="stat-val">AdamW (lr=2e-4)</span></div>
    <div class="stat-row"><span class="stat-label">Scheduler</span><span class="stat-val">ReduceLROnPlateau</span></div>

    <div class="stat-row"><span class="stat-label">Loss function</span><span class="stat-val">
      0.5 × BCE + 0.5 × Dice Loss
    </span></div>

    <div class="stat-row"><span class="stat-label">Key strategy</span><span class="stat-val">
      Weighted sampling + class imbalance handling
    </span></div>

    <div class="stat-row"><span class="stat-label">Hardware</span><span class="stat-val">Kaggle T4 × 2 GPU</span></div>
  </div>
</div>
  </div>
</div>

"""

# ═══════════════════════════════════════════════════════════════════════
# §2b — BASELINE MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════
BASELINE_RESULTS_HTML = """
<div class="section-wrap">
  <div class="section-title">§2b &mdash; Baseline Model Evaluation</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">

    <div style="padding:1.2rem;background:var(--bg-card);border:1px solid var(--border);border-radius:12px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
        <span style="font-family:var(--font-mono);font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#f59e0b;">YOLOv11n — Test Set Results</span>
      </div>
      <div style="font-size:0.76rem;color:var(--text-muted);margin-bottom:0.9rem;">Evaluated on held-out test split via Ultralytics validation pipeline. Object detection metrics — bounding box IoU matching.</div>
      <table style="width:100%;border-collapse:collapse;font-size:0.84rem;font-family:var(--font-mono);">
        <thead><tr>
          <th style="text-align:left;padding:7px 12px;color:#f59e0b;background:rgba(245,158,11,0.08);border-bottom:1px solid var(--border);">Metric</th>
          <th style="text-align:left;padding:7px 12px;color:#f59e0b;background:rgba(245,158,11,0.08);border-bottom:1px solid var(--border);">Value</th>
          <th style="text-align:left;padding:7px 12px;color:#f59e0b;background:rgba(245,158,11,0.08);border-bottom:1px solid var(--border);">Notes</th>
        </tr></thead>
        <tbody>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">Precision (P)</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.869</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">TP / (TP + FP)</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">Recall (R)</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.743</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">TP / (TP + FN)</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">mAP@0.5</td><td style="padding:7px 12px;color:#00c8ff;font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.815</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">IoU threshold = 0.5</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);">mAP@0.5:0.95</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;">0.674</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;">COCO-style multi-threshold</td></tr>
        </tbody>
      </table>
      <div style="margin-top:10px;padding:8px 10px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:6px;font-size:0.74rem;color:var(--text-muted);">Source: Ultralytics model.val() on test split &middot; IoU threshold @ 0.5</div>
    </div>

    <div style="padding:1.2rem;background:var(--bg-card);border:1px solid var(--border);border-radius:12px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
        <span style="font-family:var(--font-mono);font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#34d399;">U-Net ResNet34 — Segmentation Results</span>
      </div>
      <div style="font-size:0.76rem;color:var(--text-muted);margin-bottom:0.9rem;">Pixel-level binary segmentation evaluation at threshold = 0.5. Foreground = threat object region; background = everything else.</div>
      <table style="width:100%;border-collapse:collapse;font-size:0.84rem;font-family:var(--font-mono);">
        <thead><tr>
          <th style="text-align:left;padding:7px 12px;color:#34d399;background:rgba(52,211,153,0.08);border-bottom:1px solid var(--border);">Metric</th>
          <th style="text-align:left;padding:7px 12px;color:#34d399;background:rgba(52,211,153,0.08);border-bottom:1px solid var(--border);">Value</th>
          <th style="text-align:left;padding:7px 12px;color:#34d399;background:rgba(52,211,153,0.08);border-bottom:1px solid var(--border);">Notes</th>
        </tr></thead>
        <tbody>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">Dice Score</td><td style="padding:7px 12px;color:#00c8ff;font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.9188</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">= F1 at pixel level</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">IoU (Jaccard)</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.8888</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">Intersection / Union</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">Precision</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.9299</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">Pixel-level</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">Recall</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.9297</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">Pixel-level</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);border-bottom:1px solid rgba(30,45,74,0.4);">F1 Score</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;border-bottom:1px solid rgba(30,45,74,0.4);">0.9188</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;border-bottom:1px solid rgba(30,45,74,0.4);">≈ Dice</td></tr>
          <tr><td style="padding:7px 12px;color:var(--text-muted);">Pixel Accuracy</td><td style="padding:7px 12px;color:var(--text-primary);font-weight:700;">0.9955</td><td style="padding:7px 12px;color:var(--text-muted);font-size:0.74rem;">All pixels correct</td></tr>
        </tbody>
      </table>
      <div style="margin-top:10px;padding:8px 10px;background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.15);border-radius:6px;font-size:0.74rem;color:var(--text-muted);">Threshold = 0.5 &middot; Evaluated on validation split &middot; Pixel-level binary classification</div>
    </div>
  </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════
# §3 — ARCHITECTURE DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════
ARCH_DIAGRAMS_HTML = """
<div class="section-wrap">
  <div class="section-title">§3 &mdash; Architecture Designs</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px;">

    <div class="arch-card">
      <div class="arch-card-header"><span class="arch-num">Arch. 1</span><span class="arch-card-title">Score Refinement</span></div>
      <div class="arch-family precision">⬆ Precision-Oriented</div>
      <div style="font-size:0.77rem;color:var(--text-muted);line-height:1.5;margin-bottom:0.5rem;">For each YOLO box below HIGH_CONF (0.50), a cropped U-Net computes foreground ratio. High ratio → confidence boost (+0.06×ratio); low ratio → penalty (−0.18×(1−ratio)).</div>
      <div class="flow">
        <div class="flow-box yolo">YOLO boxes</div><div class="flow-arr">→</div>
        <div class="flow-box">conf ≥ 0.50?</div><div class="flow-arr">→</div>
        <div class="flow-box unet">Crop U-Net</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">±Δ conf</div><div class="flow-arr">→</div>
        <div class="flow-box out">Output</div>
      </div>
      <div class="arch-metrics-mini">
        <div class="mini-metric"><span class="mk">P</span><span class="mv best">0.854</span></div>
        <div class="mini-metric"><span class="mk">R</span><span class="mv">0.766</span></div>
        <div class="mini-metric"><span class="mk">mAP</span><span class="mv">0.784</span></div>
        <div class="mini-metric"><span class="mk">F1</span><span class="mv">0.807</span></div>
      </div>
    </div>

    <div class="arch-card">
      <div class="arch-card-header"><span class="arch-num">Arch. 2</span><span class="arch-card-title">Sequential Filtering</span></div>
      <div class="arch-family precision">⬆ Precision-Oriented</div>
      <div style="font-size:0.77rem;color:var(--text-muted);line-height:1.5;margin-bottom:0.5rem;">Full-image custom U-Net generates a binary mask. Each YOLO box is accepted only if its region's mask overlap ≥ 15%. Acts as a hard foreground gate.</div>
      <div class="flow">
        <div class="flow-box yolo">YOLO boxes</div><div class="flow-arr">→</div>
        <div class="flow-box unet">U-Net mask</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">overlap ≥ 15%?</div><div class="flow-arr">→</div>
        <div class="flow-box out">Keep / Drop</div>
      </div>
      <div class="arch-metrics-mini">
        <div class="mini-metric"><span class="mk">P</span><span class="mv">0.838</span></div>
        <div class="mini-metric"><span class="mk">R</span><span class="mv">0.785</span></div>
        <div class="mini-metric"><span class="mk">mAP</span><span class="mv">0.748</span></div>
        <div class="mini-metric"><span class="mk">F1</span><span class="mv best">0.810 ★</span></div>
      </div>
    </div>

    <div class="arch-card">
      <div class="arch-card-header"><span class="arch-num">Arch. 3</span><span class="arch-card-title">Parallel Score Refinement</span></div>
      <div class="arch-family precision">⬆ Precision-Oriented</div>
      <div style="font-size:0.77rem;color:var(--text-muted);line-height:1.5;margin-bottom:0.5rem;">YOLO and full-image U-Net run in parallel. Box score boosted if overlap ≥ 20% (+0.15×ov), penalised if &lt; 50% (−0.01×(1−ov)). Outputs pixel mask.</div>
      <div class="flow">
        <div class="flow-box yolo">YOLO</div><div class="flow-arr">+</div>
        <div class="flow-box unet">Full U-Net</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">Pixel overlap</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">±Δ conf</div><div class="flow-arr">→</div>
        <div class="flow-box out">+Mask</div>
      </div>
      <div class="arch-metrics-mini">
        <div class="mini-metric"><span class="mk">P</span><span class="mv">0.832</span></div>
        <div class="mini-metric"><span class="mk">R</span><span class="mv">0.781</span></div>
        <div class="mini-metric"><span class="mk">mAP</span><span class="mv">0.793</span></div>
        <div class="mini-metric"><span class="mk">F1</span><span class="mv">0.805</span></div>
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">

    <div class="arch-card">
      <div class="arch-card-header"><span class="arch-num">Arch. 4</span><span class="arch-card-title">Learned Confidence Fusion</span></div>
      <div class="arch-family precision">⬆ Precision-Oriented</div>
      <div style="font-size:0.77rem;color:var(--text-muted);line-height:1.5;margin-bottom:0.5rem;">Logistic regression trained on val-set maps 4 features (YOLO conf, fg_ratio, blob_size, mask_mean) → P(TP). Boxes below CONF_OPER dropped.</div>
      <div class="flow">
        <div class="flow-box yolo">YOLO conf</div><div class="flow-arr">+</div>
        <div class="flow-box unet">Crop feats</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">LogReg</div><div class="flow-arr">→</div>
        <div class="flow-box out">P(TP)</div>
      </div>
      <div class="arch-metrics-mini">
        <div class="mini-metric"><span class="mk">P</span><span class="mv best">0.926 ★</span></div>
        <div class="mini-metric"><span class="mk">R</span><span class="mv">0.686</span></div>
        <div class="mini-metric"><span class="mk">mAP</span><span class="mv">0.661</span></div>
        <div class="mini-metric"><span class="mk">F1</span><span class="mv">0.788</span></div>
      </div>
    </div>

    <div class="arch-card">
      <div class="arch-card-header"><span class="arch-num">Arch. 5</span><span class="arch-card-title">Attention Fusion + Injection</span></div>
      <div class="arch-family recall">⬆ Recall-Oriented</div>
      <div style="font-size:0.77rem;color:var(--text-muted);line-height:1.5;margin-bottom:0.5rem;">Sigmoid α gate (ALPHA_MIN=0.55) blends YOLO and U-Net signals. Low-conf detections pulled up by U-Net. New boxes injected from unmatched U-Net blobs.</div>
      <div class="flow">
        <div class="flow-box yolo">YOLO</div><div class="flow-arr">+</div>
        <div class="flow-box unet">Blobs</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">α·Y+(1-α)·U</div><div class="flow-arr">→</div>
        <div class="flow-box out">+Inject</div>
      </div>
      <div class="arch-metrics-mini">
        <div class="mini-metric"><span class="mk">P</span><span class="mv">0.457</span></div>
        <div class="mini-metric"><span class="mk">R</span><span class="mv best">0.845 ★</span></div>
        <div class="mini-metric"><span class="mk">mAP</span><span class="mv">0.794</span></div>
        <div class="mini-metric"><span class="mk">F1</span><span class="mv">0.593</span></div>
      </div>
    </div>

    <div class="arch-card">
      <div class="arch-card-header"><span class="arch-num">Arch. 6</span><span class="arch-card-title">Controlled Recall Recovery</span></div>
      <div class="arch-family recall">⬆ Recall-Oriented</div>
      <div style="font-size:0.77rem;color:var(--text-muted);line-height:1.5;margin-bottom:0.5rem;">Attention gating + rescue: if U-Net IoU &gt; 0.30 and YOLO conf &lt; 0.60, score boosted by 0.18 × IoU. Balances recall gain vs. precision cost.</div>
      <div class="flow">
        <div class="flow-box yolo">YOLO</div><div class="flow-arr">+</div>
        <div class="flow-box unet">IoU support</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">Rescue?</div><div class="flow-arr">→</div>
        <div class="flow-box fuse">α-blend</div><div class="flow-arr">→</div>
        <div class="flow-box out">Output</div>
      </div>
      <div class="arch-metrics-mini">
        <div class="mini-metric"><span class="mk">P</span><span class="mv">0.620</span></div>
        <div class="mini-metric"><span class="mk">R</span><span class="mv">0.826</span></div>
        <div class="mini-metric"><span class="mk">mAP</span><span class="mv best">0.796 ★</span></div>
        <div class="mini-metric"><span class="mk">F1</span><span class="mv">0.708</span></div>
      </div>
    </div>
  </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════
# §4 — FULL RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════
FINAL_TABLE_HTML = """
<div class="section-wrap" style="margin-bottom:2rem;">
  <div class="section-title">§4 &mdash; Full Results Comparison</div>
  <div style="padding:1.2rem;background:#0d1424;border:1px solid #1e2d4a;border-radius:14px" class="arch-table">
    <table>
      <thead><tr>
        <th>Architecture</th><th>Family</th><th>mAP@0.5</th><th>Precision</th><th>Recall</th><th>F1</th>
      </tr></thead>
      <tbody>
        <tr><td>YOLO Baseline</td><td style="color:var(--text-muted)">—</td><td>0.8023</td><td>0.8113</td><td>0.7887</td><td>0.7998</td></tr>
        <tr><td>Arch. 1: Score Refinement</td><td style="color:#f59e0b">Precision</td><td>0.7843</td><td>0.8539</td><td>0.7655</td><td>0.8073</td></tr>
        <tr><td>Arch. 2: Sequential Filtering</td><td style="color:#f59e0b">Precision</td><td>0.7475</td><td>0.8376</td><td>0.7848</td><td><strong style="color:#00c8ff">0.8104 ★</strong></td></tr>
        <tr><td>Arch. 3: Parallel Score Refinement</td><td style="color:#f59e0b">Precision</td><td>0.7932</td><td>0.8315</td><td>0.7805</td><td>0.8052</td></tr>
        <tr><td>Arch. 4: Learned Confidence Fusion</td><td style="color:#f59e0b">Precision</td><td>0.6605</td><td><strong style="color:#00ff9d">0.9263 ★</strong></td><td>0.6858</td><td>0.7881</td></tr>
        <tr><td>Arch. 5: Attention Fusion + Injection</td><td style="color:#34d399">Recall</td><td>0.7944</td><td>0.4566</td><td><strong style="color:#00ff9d">0.8448 ★</strong></td><td>0.5928</td></tr>
        <tr style="background:rgba(0,200,255,0.04)"><td><strong style="color:#00c8ff">Arch. 6: Controlled Recall Recovery</strong></td><td style="color:#34d399">Recall</td><td><strong style="color:#00c8ff">0.7963 ★</strong></td><td>0.6195</td><td>0.8262</td><td>0.7081</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════
# GRADIO APP
# ═══════════════════════════════════════════════════════════════════════
with gr.Blocks(
    title="X-Ray Threat Detection — Thesis Defence",
    css=CSS,
    js=MODAL_JS,
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Sans"),
    ),
) as demo:

    gr.HTML(HEADER_HTML)

    # ── Detection interface ───────────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=290, elem_classes=["panel"]):
            gr.HTML('<div class="panel-label" style="color:#7a8ba8;margin-bottom:8px">INPUT IMAGE</div>')
            img_input = gr.Image(type="pil", label="Upload X-ray image",
                                 elem_classes=["upload-box"], height=260)
            gr.HTML('<div class="panel-label" style="color:#7a8ba8;margin:14px 0 8px">SELECT ARCHITECTURE</div>')
            arch_radio = gr.Radio(choices=ARCH_NAMES, value=ARCH_NAMES[0],
                                  label="", elem_classes=["arch-radio"])
            run_btn = gr.Button("▶  Run Detection", variant="primary",
                                size="lg", elem_classes=["run-btn"])
            gr.HTML("""<div class="legend">
              <strong style="color:#e8eef8">Legend</strong><br>
              <span style="color:#ff9500">■</span> YOLO Baseline<br>
              <span style="color:#00ff9d">■</span> U-Net Mask<br>
              <span style="color:#00c8ff">■</span> Hybrid Result<br>
            </div>""")

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="panel-label yolo">① YOLO Baseline</div>')
                    out_yolo = gr.Image(label="", show_label=False,
                                       elem_classes=["output-image"], height=380)
                with gr.Column():
                    gr.HTML('<div class="panel-label mask">② U-Net Mask</div>')
                    out_mask = gr.Image(label="", show_label=False,
                                       elem_classes=["output-image"], height=380)
                with gr.Column():
                    gr.HTML('<div class="panel-label hybrid">③ Hybrid Result</div>')
                    out_hybrid = gr.Image(label="", show_label=False,
                                         elem_classes=["output-image"], height=380)
            out_text = gr.Markdown(
                value="*Upload an X-ray image and press **▶ Run Detection** to start.*",
                elem_classes=["metric-md"]
            )

    # ── Pipeline showcase sections ────────────────────────────────────
    gr.HTML(DATA_PREP_HTML)
    gr.HTML(TRAINING_HTML)
    gr.HTML(BASELINE_RESULTS_HTML)
    gr.HTML(ARCH_DIAGRAMS_HTML)
    gr.HTML(FINAL_TABLE_HTML)

    run_btn.click(
        fn=predict,
        inputs=[img_input, arch_radio],
        outputs=[out_yolo, out_mask, out_hybrid, out_text]
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["."])