import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from eval_asl_citizen_retrieval import load_metadata, load_features

GLOSSES=["VIOLIN","CAMERA","TREE","CAR","SCISSORS","BUTTERFLY","TOOTHBRUSH","SMOKING"]
recs=load_metadata(Path("/home/psobecki/ASL_Citizen/splits"),"train")
feats,glosses=load_features(recs,Path("/scratch-shared/psobecki/ASL_Citizen/logos_features_native"),None,desc="load raw",workers=16,min_coverage=0,allow_partial=True)
feats=F.normalize(torch.from_numpy(feats).float(),dim=-1).numpy()
glosses=np.array(glosses)
present=[g for g in GLOSSES if (glosses==g).any()]
print("present:",present)
rng=np.random.default_rng(0); X=[];L=[]
for g in present:
    idx=np.where(glosses==g)[0]
    take=rng.choice(idx,size=min(14,len(idx)),replace=False)
    X.append(feats[take]); L+=[g]*len(take)
X=np.concatenate(X); L=np.array(L)
proj=TSNE(n_components=2,perplexity=15,init="pca",random_state=0).fit_transform(X)
fig,ax=plt.subplots(figsize=(6.5,5.5)); cmap=plt.get_cmap("tab10"); mk=["s","^","o","D","v","P","X","*"]
for i,g in enumerate(present):
    m=L==g
    ax.scatter(proj[m,0],proj[m,1],color=cmap(i%10),marker=mk[i%len(mk)],s=30,alpha=0.6,label=g)
    ax.scatter([proj[m,0].mean()],[proj[m,1].mean()],color=cmap(i%10),marker=mk[i%len(mk)],s=250,edgecolors="black",linewidths=1.5,zorder=5)
ax.legend(fontsize=8); ax.set_title("raw Logos: high-iconicity glosses, different meanings (t-SNE)")
fig.tight_layout(); fig.savefig("runs/king_queen_analogy/iconicity_tsne_raw.png",dpi=150); print("saved iconicity_tsne_raw.png")
