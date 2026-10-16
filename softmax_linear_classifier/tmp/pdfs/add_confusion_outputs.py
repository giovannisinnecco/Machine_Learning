import base64
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd()
items=[('1a','model_training_1a','softmax','Softmax | 561 atributos extraidos'),('1b','model_traing_1b','knn','KNN | 561 atributos extraidos'),('2a','model_traing_2a','softmax','Softmax | 768 valores ACC/GYR'),('2b','model_traing_2b','knn','KNN | 768 valores ACC/GYR')]
for item,nbname,model,title in items:
 path=root/f'notebooks/{nbname}.ipynb'
 nb=json.loads(path.read_text(encoding='utf-8'))
 source='''import matplotlib.pyplot as plt
import numpy as np

# Linhas: classes reais. Colunas: classes previstas. Valores absolutos.
rotulos = ["Caminhar", "Subir\\nescadas", "Descer\\nescadas", "Sentado", "Em pé", "Deitado"]
matriz = np.asarray(matriz_confusao_teste, dtype=int)
fig, ax = plt.subplots(figsize=(9, 7.6), layout="constrained")
imagem = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=537)
barra = fig.colorbar(imagem, ax=ax, shrink=0.82, pad=0.03)
barra.set_label("Número de amostras")
ax.set_xticks(range(6), labels=rotulos)
ax.set_yticks(range(6), labels=[r.replace("\\n", " ") for r in rotulos])
ax.set_xlabel("Classe prevista", fontsize=12, labelpad=12)
ax.set_ylabel("Classe real", fontsize=12, labelpad=12)
ax.set_title("ITEM_TITLE\\nMatriz de confusão no teste oficial (2.947 amostras)", fontsize=14, pad=18)
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                fontsize=12, color="white" if matriz[i, j] > 537 / 2 else "#18334A")
ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)
acertos = int(np.trace(matriz))
fig.supxlabel(
    f"Acertos: {acertos}/2947  |  Acurácia: {100 * metricas_finais['acuracia_teste']:.2f}%"
    f"  |  Balanceada: {100 * metricas_finais['acuracia_balanceada_teste']:.2f}%", fontsize=11,
)
pasta_figuras = project_root / "results" / "figures"
pasta_figuras.mkdir(parents=True, exist_ok=True)
fig.savefig(pasta_figuras / "confusao_ITEM.png", dpi=180, facecolor="white")
plt.show()
'''.replace('ITEM_TITLE',item+' - '+title).replace('confusao_ITEM',f'confusao_{item}')
 metrics=json.loads((root/f'results/{model}_{item}_metricas.json').read_text())
 ns={'project_root':root,'matriz_confusao_teste':np.asarray(metrics['matriz_confusao_teste']),'metricas_finais':metrics}
 exec(compile(source,f'confusao_{item}','exec'),ns)
 png=root/f'results/figures/confusao_{item}.png'
 count=max([c.get('execution_count') or 0 for c in nb['cells']])+1
 cell={'cell_type':'code','execution_count':count,'id':f'confusion-matrix-{item}','metadata':{},'source':source.splitlines(keepends=True),'outputs':[{'output_type':'display_data','data':{'image/png':base64.b64encode(png.read_bytes()).decode('ascii'),'text/plain':['Matriz de confusao - '+item]},'metadata':{}}]}
 nb['cells']=[c for c in nb['cells'] if c.get('id')!=cell['id']]+[cell]
 path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
 assert np.asarray(metrics['matriz_confusao_teste']).sum()==2947
 plt.close('all')
 print(png)
