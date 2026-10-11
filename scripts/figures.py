from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.audit import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':160})
p=ROOT/'results';figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
f=pd.read_csv(p/'token_counts.csv');r=pd.read_csv(p/'replays.csv');l=pd.read_csv(p/'language_results.csv');fr=pd.read_csv(p/'frontiers.csv');b=pd.read_csv(p/'bootstrap.csv')
TOKS=['cl100k_base','o200k_base','qwen2.5'];TITLES=['cl100k_base','o200k_base','Qwen2.5'];C={'chars4':'#d55e00','exact':'#0072b2','calibrated':'#009e73','bytes':'#777777'}
# Two-ledger pipeline diagram
fig,ax=plt.subplots(figsize=(7.2,2.1));ax.axis('off')
boxes=[(.01,.48,.2,.33,'Translated task\n(same sentence ID)'),(.28,.48,.2,.33,'Reserve estimate r\ncheck counter + r ≤ B'),(.55,.48,.18,.33,'Admit / reject\nno retries'),(.79,.48,.2,.33,'Settle true count c\nadd c − r')]
from matplotlib.patches import FancyBboxPatch
for x,y,w,h,t in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',facecolor='#eef3f7',edgecolor='#405568'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
for a,bb in zip(boxes,boxes[1:]):ax.annotate('',xy=(bb[0]-.01,.645),xytext=(a[0]+a[2]+.01,.645),arrowprops={'arrowstyle':'->','color':'#405568'})
ax.text(.38,.15,'Reservation ledger: decisions use r',color=C['exact'],ha='center');ax.text(.81,.15,'Usage ledger: obligations use c',color=C['chars4'],ha='center')
fig.savefig(figdir/'pipeline.pdf',bbox_inches='tight');fig.savefig(figdir/'pipeline.png',bbox_inches='tight');plt.close(fig)
# full language coverage heatmaps
fig,axs=plt.subplots(1,3,figsize=(7.2,4),sharey=True)
for ax,tok,title in zip(axs,TOKS,TITLES):
 mat=l[l.tokenizer==tok].groupby(['language','policy']).coverage.mean().unstack().loc[LANGS,['chars4','exact','calibrated']].to_numpy()
 im=ax.imshow(mat,vmin=0,vmax=1,cmap='cividis',aspect='auto')
 for i in range(12):
  for j in range(3):ax.text(j,i,f'{100*mat[i,j]:.1f}',ha='center',va='center',color='white' if mat[i,j]<.5 else 'black',fontsize=7)
 ax.set_xticks([0,1,2],['Chars/4','Exact','Calibrated'],rotation=25,ha='right');ax.set_yticks(range(12),LABELS);ax.set_title(title);ax.tick_params(length=0)
fig.subplots_adjust(bottom=.2,wspace=.12,right=.86)
cax=fig.add_axes([.89,.24,.018,.6])
fig.colorbar(im,cax=cax,label='Admitted fraction')
fig.savefig(figdir/'admission.pdf',bbox_inches='tight');fig.savefig(figdir/'admission.png',bbox_inches='tight');plt.close(fig)
# estimator error
fig,axs=plt.subplots(1,2,figsize=(7.2,2.7))
d=f[f.split=='devtest'].groupby('language')[TOKS+['chars4']].mean().loc[LANGS]
for tok,title,marker in zip(TOKS,TITLES,['o','s','^']):
 axs[0].plot(range(12),d[tok]/d['chars4'],marker=marker,label=title,lw=1.2,ms=4)
axs[0].axhline(1,color='black',linestyle='--',lw=.8);axs[0].set_ylabel('True tokens / reservation');axs[0].set_xticks(range(12),LABELS,rotation=65,ha='right');axs[0].legend(fontsize=7)
for tok,title,marker in zip(TOKS,TITLES,['o','s','^']):
 g=r.query('fraction==.5 and policy=="chars4" and tokenizer==@tok').groupby('batch').budget_ratio.mean()
 axs[1].plot(g.index,g.values,marker=marker,label=title)
axs[1].set_xscale('log');axs[1].set_xticks([1,8,64,1012],[1,8,64,1012]);axs[1].set_xlabel('Requests before settlement');axs[1].set_ylabel('Committed tokens / budget');axs[1].axhline(1,color='black',linestyle='--',lw=.8)
fig.tight_layout();fig.savefig(figdir/'ledger_error.pdf',bbox_inches='tight');fig.savefig(figdir/'ledger_error.png',bbox_inches='tight');plt.close(fig)
# frontier plot with no cherry-picking hidden
fig,axs=plt.subplots(1,3,figsize=(7.2,2.65),sharey=True)
for ax,tok,title in zip(axs,TOKS,TITLES):
 for pol,mark in [('exact','o'),('calibrated','s')]:
  g=r.query('tokenizer==@tok and policy==@pol and batch==1012').groupby('fraction').min_coverage.mean()
  ax.plot(g.index,g.values,color=C[pol],marker=mark,label='Equal tokens' if pol=='exact' else 'Calibrated')
 g=fr[fr.tokenizer==tok].groupby('fraction')[['prefix','cherry']].mean()
 ax.plot(g.index,g.prefix,color='black',linestyle='--',label='Paired prefix')
 ax.plot(g.index,g.cherry,color='#777777',linestyle=':',label='Cheapest-count bound')
 ax.set_title(title);ax.set_xlabel('Quota / English token demand');ax.set_xticks([.25,.5,.75]);ax.set_ylim(0,.72)
axs[0].set_ylabel('Minimum admitted fraction');axs[1].legend(fontsize=6,loc='upper left');fig.tight_layout()
fig.savefig(figdir/'frontier.pdf',bbox_inches='tight');fig.savefig(figdir/'frontier.png',bbox_inches='tight');plt.close(fig)
# Generated manuscript tables, never hand-copy numeric values.
with (ROOT/'paper/results_table.tex').open('w') as out:
 out.write('\\begin{tabular}{llrrrr}\\toprule\nTokenizer & Policy & Admitted (\\%) & Gap (pp) & Usage / budget & Max. key ratio\\\\\\midrule\n')
 for tok,title in zip(TOKS,TITLES):
  latex_title=title.replace("_",chr(92)+"_")
  g=r.query('tokenizer==@tok and fraction==.5 and batch==1012').groupby('policy').mean(numeric_only=True)
  for pol in ['chars4','exact','bytes','calibrated']:
   x=g.loc[pol];out.write(f'{latex_title if pol=="chars4" else ""} & {pol.capitalize()} & {100*x.admission:.2f} & {100*x.gap:.2f} & {x.budget_ratio:.3f} & {x.max_key_ratio:.3f}\\\\\n')
  if tok!=TOKS[-1]:out.write('\\midrule\n')
 out.write('\\bottomrule\\end{tabular}\n')
with (ROOT/'paper/bootstrap_table.tex').open('w') as out:
 out.write('\\begin{tabular}{lrrr}\\toprule\nTokenizer & Gap reduction (pp) & Throughput ratio & Minimum access (\\%)\\\\\\midrule\n')
 for tok,title in zip(TOKS,TITLES):
  latex_title=title.replace("_",chr(92)+"_")
  g=b[b.tokenizer==tok];q=g[['gap_reduction','throughput_ratio','min_cal']].quantile([.008333333,.991666667])
  out.write(f'{latex_title} & [{100*q.gap_reduction.iloc[0]:.2f}, {100*q.gap_reduction.iloc[1]:.2f}] & [{q.throughput_ratio.iloc[0]:.3f}, {q.throughput_ratio.iloc[1]:.3f}] & [{100*q.min_cal.iloc[0]:.2f}, {100*q.min_cal.iloc[1]:.2f}]\\\\\n')
 out.write('\\bottomrule\\end{tabular}\n')
summary={'main_rows':len(r),'main_decisions':len(r)*1012*12,'token_counts':len(f)*3,'texts':len(f)}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
