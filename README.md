## Ressourcen
[Conditional Autoregressive Models in Stan](https://github.com/mbjoseph/CARstan). Hiervon hatte ich auch einiges für die Präsi verwendet.

[Stan Documentation über censored data](https://mc-stan.org/docs/2_18/stan-users-guide/truncated-or-censored-data.html). Das will ich mit ein paar Testdaten implementieren. Oder trifft _truncated data_ eher auf's Tobit model zu?


## Anleitung

Moin - installiert bitte [Anaconda](https://www.anaconda.com/distribution/) bzw [Miniconda](https://docs.conda.io/en/latest/miniconda.html) wenn ihr's noch nicht habt. Anschließend cloned ihr mein Repo dorthin, wo ihr es haben wollt, und richtet es dann ein:

```
git clone https://github.com/LyteFM/spatial-tobit.git
# bzw mit ssh keys, wenn ihr's eingerichtet habt
# git clone git@github.com:LyteFM/spatial-tobit.git
cd spatial-tobit
```

---

Vanessa - Du kannst deine Umgebung so einrichten:

```bash
conda create --name probmod --file spec-file.txt
source activate probmod
pip install pystan 
```

Christian - Bei dir geht's nur manuell:

```
conda create -n probmod python=3.7 anaconda
(source) activate probmod # on windows, source prefix might not work
conda install libpython m2w64-toolchain -c msys2
conda install numpy cython matplotlib scipy pandas -c conda-forge -y
pip install pystan
conda clean -p -y # free some space after change to conda-forge
```

Und falls irgendwas nicht klappt, hier schauen:

[https://pystan.readthedocs.io/en/latest/windows.html#windows](https://pystan.readthedocs.io/en/latest/windows.html#windows)

---

Anschließend innerhalb der Umgebung (sollte `(probmod)` noch sichtbar sein) testen, ob die Installation geklappt hat:

`python pystan_example.py`

Die Umgebung lassen wir dann erstmal so und sollten eigentlich keine komischen Überraschungen durch unterschiedliche Versionen bekommen. Müssten auch alle Pakete zum Data Cleaning, Visualisieren etc dabei sein. Programme wie Spyder/ Jupyter Notebook könnt ihr natürlich auch wieder deinstallieren / aus dem Spec-File rausnehmen, wenn ihr die nicht verwendet.
