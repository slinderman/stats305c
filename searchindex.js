Search.setIndex({"docnames": ["assignments/hw0/hw0", "assignments/hw1/hw1", "assignments/hw2/hw2", "assignments/hw3/hw3", "assignments/hw4/hw4", "index", "lectures/99_references", "notebooks/01_bayes_normal", "notebooks/02_mvn", "notebooks/03_hier_gauss", "notebooks/04_mcmc"], "filenames": ["assignments/hw0/hw0.ipynb", "assignments/hw1/hw1.ipynb", "assignments/hw2/hw2.ipynb", "assignments/hw3/hw3.ipynb", "assignments/hw4/hw4.ipynb", "index.md", "lectures/99_references.md", "notebooks/01_bayes_normal.ipynb", "notebooks/02_mvn.ipynb", "notebooks/03_hier_gauss.ipynb", "notebooks/04_mcmc.ipynb"], "titles": ["HW0: PyTorch Primer", "HW1: Bayesian Linear Regression", "HW2: Gibbs Sampling and Metropolis-Hastings", "HW3: Continuous Latent Variable Models", "HW4: Bayesian Mixture Models", "Overview", "References", "Bayesian Analysis of the Normal Distribution", "The Multivariate Normal Distribution", "Hierarchical Gaussian Models", "Markov Chain Monte Carlo"], "terms": {"we": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], "ll": [0, 1, 3, 4, 7, 8, 10], "us": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], "python": [0, 1, 3, 5, 7, 8, 10], "assign": [0, 1, 2, 3, 4, 5], "thi": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], "cours": [0, 8], "lab": 0, "help": [0, 3, 5, 8, 9], "you": [0, 1, 2, 3, 4, 5, 8, 9], "get": [0, 1, 2, 3, 4, 9], "up": [0, 1, 2, 4, 5], "speed": [0, 2], "It": [0, 1, 2, 3, 4, 7, 8], "introduc": [0, 2, 8, 9], "s": [0, 1, 2, 3, 4, 7, 9, 10], "equival": [0, 1, 3], "numpi": [0, 3, 4, 7, 9], "arrai": [0, 3, 7, 9], "more": [0, 1, 4, 7, 8, 9, 10], "bell": 0, "whistl": 0, "run": [0, 1, 2, 10], "gpu": [0, 3], "support": [0, 1, 2, 3], "automat": 0, "differenti": [0, 5], "If": [0, 1, 2, 3, 4, 7], "re": [0, 1, 2, 3, 4, 9], "come": [0, 1], "from": [0, 1, 3, 4, 7, 10], "matlab": [0, 3], "probabl": [0, 1, 3, 5, 9], "know": [0, 1, 2, 7], "can": [0, 1, 2, 4, 7, 8, 9, 10], "avoid": 0, "costli": 0, "loop": 0, "comput": [0, 2, 3, 4, 8, 10], "over": [0, 1, 2, 3, 4, 8, 10], "dimens": [0, 3, 4, 8], "an": [0, 1, 2, 3, 4, 7, 8], "here": [0, 1, 2, 3, 4, 7, 9, 10], "trick": [0, 3, 9, 10], "ha": [0, 1, 2, 3, 4, 9, 10], "excel": 0, "librari": [0, 9], "sampl": [0, 4, 7], "evalu": [0, 3, 7, 9], "log": [0, 3, 9], "much": [0, 10], "import": [0, 1, 2, 3, 4, 7, 8, 9, 10], "torch": [0, 1, 2, 3, 4, 7, 8, 9, 10], "dist": [0, 3], "matplotlib": [0, 1, 2, 3, 4, 7, 8, 9, 10], "pyplot": [0, 1, 2, 3, 4, 7, 8, 9, 10], "plt": [0, 1, 2, 3, 4, 7, 8, 9, 10], "ar": [0, 1, 2, 3, 4, 7, 8, 9, 10], "The": [0, 2, 4, 7, 9, 10], "document": 0, "alreadi": 0, "great": 0, "tutori": [0, 1], "rather": [0, 3, 9], "than": [0, 3, 8, 9], "recreat": 0, "wheel": 0, "pleas": [0, 1, 3, 4, 10], "start": [0, 3, 8], "read": [0, 5], "onc": [0, 3, 9], "ve": [0, 1, 2, 3, 4], "through": [0, 1, 3, 9, 10], "try": [0, 1, 3], "function": [0, 2, 7, 9, 10], "like": [0, 1, 2, 3, 4, 8, 9, 10], "arang": [0, 3, 9, 10], "reshap": [0, 3, 4], "etc": 0, "follow": [0, 1, 2, 3, 4, 5, 7, 9], "0": [0, 1, 2, 3, 4, 7, 8, 9, 10], "7": [0, 3, 9, 10], "8": [0, 3, 4, 5, 7, 8, 9, 10], "note": [0, 1, 2, 3, 7, 8, 9, 10], "For": [0, 1, 3, 4, 7, 8], "ones": [0, 3, 4, 10], "below": [0, 1, 2, 3, 4], "don": [0, 1], "t": [0, 1, 2, 3, 4, 7, 8, 9], "liter": 0, "specifi": [0, 2, 3, 4, 10], "list": [0, 1, 3, 10], "your": [0, 1, 2, 7], "code": [0, 5, 9], "sequenc": 0, "repeat": [0, 3, 4], "time": [0, 1, 2, 3, 4, 5, 8, 10], "doe": [0, 1, 2, 4, 7, 8], "arbitrari": 0, "number": [0, 2, 3, 4, 9, 10], "life": 0, "easier": 0, "faster": 0, "hang": 0, "x": [0, 1, 3, 4, 7, 8, 9, 10], "where": [0, 1, 3, 4, 7, 8, 9, 10], "i": [0, 1, 2, 3, 4, 7, 8, 9, 10], "j": [0, 3, 8], "sum": [0, 1, 3, 4, 7, 8, 9, 10], "two": [0, 4, 8], "dimension": [0, 3, 4, 5, 8, 9], "exampl": [0, 3, 4, 7, 8, 10], "distanc": [0, 3], "matrix": [0, 1, 4, 5, 8], "d": [0, 1, 3, 4, 7, 8, 9], "euclidean": 0, "between": [0, 1, 2, 3, 4, 9, 10], "10": [0, 1, 2, 3, 5, 7, 8, 9, 10], "dtype": [0, 2], "float": [0, 4], "answer": [0, 1, 2], "should": [0, 1, 2, 3, 4, 5, 9], "0000": 0, "8284": 0, "6569": 0, "4853": 0, "11": [0, 5, 9, 10], "3137": 0, "extract": 0, "submatrix": 0, "row": [0, 1, 3, 4], "column": [0, 1, 2, 3, 4], "A": [0, 4, 8, 9, 10], "25": [0, 2, 3, 5, 7, 9, 10], "14": [0, 9, 10], "15": [0, 2, 9, 10], "16": [0, 3, 5, 7, 8, 9, 10], "19": [0, 10], "creat": [0, 1, 4, 7, 9], "binari": [0, 1, 2, 3], "mask": 0, "m": [0, 3, 4], "same": [0, 1, 2, 3, 10], "shape": [0, 1, 2, 3, 4, 7, 8, 10], "true": [0, 1, 3, 7, 8, 9, 10], "onli": [0, 1, 2, 3, 4, 9], "divis": 0, "let": [0, 1, 2, 3, 4, 7, 8, 9, 10], "fals": [0, 3], "add": [0, 3, 4, 10], "one": [0, 2, 3, 4, 7, 8, 9, 10], "entri": [0, 3], "after": [0, 2, 7, 10], "updat": [0, 2, 3, 8], "place": [0, 1, 3], "9": [0, 5, 7, 8, 9, 10], "12": [0, 4, 5, 7, 9, 10], "13": [0, 5, 9, 10], "17": [0, 2, 9, 10], "18": [0, 5, 9, 10], "20": [0, 1, 2, 5, 7, 10], "22": [0, 10], "23": [0, 5, 10], "24": [0, 10], "doc": [0, 1], "object": [0, 1, 3, 4, 7, 8], "fit": [0, 3, 4], "poisson": [0, 5], "mixtur": [0, 5], "model": [0, 1, 5], "draw": [0, 1, 2, 4, 7, 10], "50": [0, 1, 3, 7, 8, 9, 10], "rate": [0, 2, 7], "One": [0, 1], "awesom": 0, "thing": [0, 3, 7, 10], "about": [0, 1, 2, 3, 4, 9], "thei": [0, 3, 7, 9, 10], "too": 0, "p": [0, 1, 2, 3, 7, 8, 9, 10], "equal": [0, 1, 3], "mathrm": [0, 2, 3, 4, 7, 8, 9], "poi": 0, "lambda": [0, 1, 7, 8], "ldot": [0, 2, 3, 7, 8, 9, 10], "3679": 0, "1839": 0, "0613": 0, "0153": 0, "1353": 0, "2707": 0, "1804": 0, "0902": 0, "0498": 0, "1494": 0, "2240": 0, "1680": 0, "0183": 0, "0733": 0, "1465": 0, "1954": 0, "point": [0, 1, 2, 4, 10], "under": [0, 8, 9], "gamma": [0, 1, 2, 3, 7, 9, 10], "aka": [0, 3], "concentr": [0, 4, 9, 10], "invers": [0, 1, 3, 4, 7, 9, 10], "scale": [0, 1, 3, 4, 7, 9, 10], "0336": 0, "5905": 0, "0540": 0, "1000": [0, 9, 10], "begin": [0, 1, 2, 3, 4, 7, 8, 9, 10], "align": [0, 1, 2, 3, 4, 7, 8, 9, 10], "frac": [0, 2, 3, 4, 7, 8, 9, 10], "end": [0, 1, 2, 3, 4, 7, 8, 9, 10], "hist": [0, 7, 9, 10], "plot": [0, 2, 4, 9], "normal": [0, 1, 2, 4, 5, 9, 10], "histogram": [0, 7, 10], "data": [0, 4, 5, 8, 9], "batch": [0, 4], "100": [0, 4, 7, 8, 9, 10], "independ": [0, 1, 3, 7, 8, 9, 10], "standard": [0, 1, 2, 3, 4, 7, 8, 9, 10], "random": [0, 1, 2, 3, 4, 7, 8], "variabl": [0, 1, 2, 4, 7, 8, 9, 10], "varianc": [0, 3, 4, 10], "mean": [0, 2, 3, 4, 10], "stats305c": [1, 4], "stanford": [1, 5], "univers": [1, 5], "spring": [1, 5], "2023": [1, 5, 10], "name": [1, 2, 3, 4], "collabor": [1, 2, 3, 4], "hour": [1, 5], "spent": 1, "how": [1, 2, 3, 4, 7, 8, 9, 10], "mani": [1, 9], "total": [1, 2], "so": [1, 2, 4, 5, 9, 10], "calibr": 1, "futur": 1, "feedback": 1, "alwai": 1, "welcom": 1, "setup": 1, "transformeddistribut": [1, 3, 7, 8, 9, 10], "multivariatenorm": [1, 3, 4, 8], "transform": [1, 3, 4, 7, 8, 9, 10], "powertransform": [1, 3, 7, 9, 10], "seaborn": [1, 2, 3, 7, 8, 9, 10], "sn": [1, 2, 3, 7, 8, 9, 10], "set_context": [1, 2, 3, 7, 8, 9, 10], "notebook": [1, 2, 3, 4, 5, 7, 8, 9, 10], "stat": [1, 10], "305a": 1, "wa": [1, 2, 3, 8, 9], "all": [1, 2, 3, 4, 7, 8, 9], "In": [1, 2, 3, 4, 8, 10], "revisit": 1, "classic": [1, 9], "perspect": 1, "mathbf": [1, 3, 4, 8, 9, 10], "_n": [1, 3, 8], "y_n": 1, "_": [1, 2, 3, 4, 7, 9, 10], "n": [1, 2, 3, 4, 7, 8, 9, 10], "denot": [1, 2, 3, 7, 9, 10], "dataset": [1, 9], "covari": [1, 3, 4], "mathbb": [1, 3, 4, 7, 8, 9, 10], "r": [1, 3, 4, 7, 8, 9, 10], "scalar": [1, 2, 3, 4, 7, 10], "outcom": [1, 9], "design": [1, 9], "each": [1, 2, 9, 10], "vector": [1, 3, 4, 7, 8], "y": [1, 3, 4], "condition": [1, 7, 9, 10], "gaussian": [1, 3, 5, 7, 8], "given": [1, 2, 3, 8, 10], "paramet": [1, 2, 3, 4, 7, 8, 9, 10], "mid": [1, 2, 3, 4, 7, 8, 9, 10], "w": [1, 2, 3, 4, 7, 8], "sigma": [1, 3, 4, 7, 8, 9, 10], "prod_": [1, 2, 3, 7, 8, 9, 10], "mathcal": [1, 2, 3, 7, 8, 9, 10], "top": [1, 3, 4, 8], "weight": [1, 4, 9, 10], "condit": [1, 9, 10], "prior": [1, 2, 4, 7, 8, 9, 10], "boldsymbol": [1, 3, 8, 9, 10], "eta": [1, 3, 7, 8, 9, 10], "chi": [1, 3, 7, 9, 10], "nu_0": [1, 3, 4, 7, 8, 9, 10], "sigma_0": [1, 3, 4, 7, 10], "mu": [1, 3, 4, 7, 8, 9], "_0": [1, 8], "set": [1, 2, 3, 4, 7, 8, 10], "degre": [1, 3, 4, 7, 9, 10], "freedom": [1, 3, 4, 7, 9, 10], "_d": [1, 3], "posit": [1, 3], "definit": [1, 3], "precis": [1, 4, 9, 10], "collect": [1, 3, 10], "hyperparamet": [1, 2, 3, 4, 7, 8, 10], "As": [1, 2, 3], "class": [1, 3, 4, 7, 9, 10], "complet": [1, 4, 5, 7, 10], "portion": 1, "unfamiliar": 1, "webpag": 1, "provid": [1, 2, 4], "introductori": 1, "tensor": [1, 2, 4, 7, 8, 9, 10], "also": [1, 2, 3, 4, 5, 7, 8, 9], "make": [1, 2, 4, 8, 9], "sure": [1, 2, 3, 4], "solv": [1, 3, 4], "homework": [1, 2, 3, 4], "form": [1, 2, 3, 8, 10], "nu_n": [1, 8, 10], "sigma_n": [1, 7, 10], "some": [1, 2, 4, 9, 10], "hint": [1, 2, 3, 4], "rememb": [1, 2, 10], "procedur": [1, 4], "write": [1, 4, 10], "down": [1, 2], "joint": [1, 2, 3, 8], "both": [1, 2, 7, 8, 9], "fouc": 1, "term": [1, 2, 4, 5, 9], "care": [1, 3], "But": 1, "have": [1, 2, 3, 4, 7, 8, 9, 10], "veri": [1, 2, 8, 9], "keep": 1, "becaus": [1, 2, 9], "ask": 1, "what": [1, 2, 3, 8], "e": [1, 2, 3, 4, 7, 8, 9, 10], "uninform": 1, "limit": [1, 2, 3, 4], "b": [1, 4], "hat": [1, 9, 10], "h": [1, 3, 4], "do": [1, 2, 3, 4, 7, 8, 9], "simpl": [1, 2, 4, 8, 10], "analysi": [1, 5], "x_n": [1, 7, 8], "look": [1, 3, 8, 9, 10], "wget": [1, 4], "nc": [1, 4], "http": [1, 3, 4, 10], "raw": [1, 2, 4], "githubusercont": [1, 4], "com": [1, 4], "slinderman": [1, 4], "main": [1, 4], "pt": 1, "load": [1, 4], "x_1": [1, 8], "x_2": [1, 8], "y_1": 1, "ko": [1, 8], "xlabel": [1, 3, 4, 7, 8, 9, 10], "ylabel": [1, 3, 4, 7, 8, 9, 10], "were": [1, 3, 9], "simul": [1, 2, 7], "nois": [1, 3], "accord": [1, 2, 4], "visual": [1, 10], "mai": [1, 2, 3, 4, 5], "find": [1, 2, 3, 4, 5, 8], "command": [1, 2, 3, 4, 10], "tupl": [1, 4], "contain": [1, 2], "return": [1, 2, 3, 4, 7, 8, 9, 10], "transpos": 1, "linalg": [1, 3, 4, 8], "element": [1, 4], "wise": 1, "multipl": [1, 9], "while": 1, "def": [1, 2, 3, 4, 7, 8, 9, 10], "compute_posterior": 1, "sigmasq_0": [1, 10], "mu_0": [1, 4, 7, 8, 9, 10], "lambda_0": [1, 3, 7, 8, 9], "sigmasq_n": [1, 7, 10], "mu_n": [1, 7, 8], "lambda_n": [1, 8], "arg": [1, 2, 3, 4, 9, 10], "print": [1, 2, 3, 4, 7, 9, 10], "test": [1, 10], "hyper": [1, 2, 4, 7, 9, 10], "dict": [1, 3, 10], "zero": [1, 2, 3, 4, 8], "ey": [1, 4, 8], "vs": 1, "interv": 1, "continu": 1, "download": [1, 2, 4], "scaledinvchisq": [1, 3, 7, 9, 10], "which": [1, 2, 4, 9], "copi": [1, 4], "demo": [1, 8], "lectur": [1, 3, 9, 10], "dir": 1, "attribut": 1, "To": [1, 2, 3, 4, 9, 10], "learn": [1, 3, 4, 5], "see": [1, 2, 3, 4, 9, 10], "__init__": [1, 3, 7, 9, 10], "self": [1, 3, 7, 9, 10], "dof": [1, 3, 7, 9, 10], "implement": [1, 5, 7, 8, 9, 10], "thu": [1, 2, 7], "inherit": [1, 7], "log_prob": [1, 3, 7, 8, 9, 10], "its": [1, 2, 3, 4, 9, 10], "parent": 1, "base": [1, 2, 3, 7, 9, 10], "margin": [1, 10], "expect": [1, 2, 5, 7], "valu": [1, 2, 3, 4, 9, 10], "grid": [1, 3, 7, 9], "evenli": 1, "space": [1, 4, 5, 8, 9], "our": [1, 2, 3, 4, 7, 9, 10], "defin": [1, 3, 4, 7, 8, 9, 10], "line": [1, 2, 3, 4, 7], "other": [1, 3, 8, 9, 10], "sens": 1, "uncertainti": [1, 2], "want": [1, 3, 4], "transpar": 1, "alpha": [1, 4, 7, 8, 9, 10], "overlai": 1, "observ": [1, 2, 3, 9], "gener": [1, 2, 3, 4], "depend": [1, 3, 4, 9], "subpart": 1, "walk": 1, "new": [1, 2, 3, 4, 5, 7, 8, 9, 10], "input": [1, 4], "That": [1, 2, 10], "y_": 1, "x_": [1, 3, 9, 10], "integr": [1, 9], "found": [1, 2, 7, 10], "purpos": [1, 3], "question": [1, 4], "enough": 1, "leav": 1, "need": [1, 2, 4, 8, 9], "plug": [1, 9], "product": [1, 3, 4], "rule": [1, 9], "out": [1, 3], "next": [1, 2, 4, 8], "replac": 1, "ani": [1, 4, 9], "known": [1, 2, 7, 8, 9], "famili": 1, "notat": [1, 4, 7, 10], "text": [1, 2, 3, 4, 5, 7, 8, 9, 10], "symbol": 1, "abov": [1, 2, 3, 4, 8, 9], "now": [1, 3, 4, 7, 9, 10], "int": [1, 2, 7, 9], "without": 1, "take": [1, 2, 3, 4, 7, 10], "think": [1, 2], "conjug": [1, 2, 3, 7, 8], "student": [1, 5, 7, 9, 10], "came": 1, "format": [1, 2, 3, 4, 7, 9, 10], "check": [1, 2, 3, 4, 7], "exce": [1, 2, 3, 4], "80": [1, 2, 3, 4, 10], "charact": [1, 2, 3, 4], "width": [1, 2, 3, 4, 9], "tool": [1, 2, 3, 4], "editor": [1, 2, 3, 4], "vertic": [1, 2, 3, 4], "ruler": [1, 2, 3, 4], "when": [1, 2, 3, 4, 9], "exceed": [1, 2, 3, 4], "ipynb": [1, 2, 3, 4], "convert": [1, 2, 3, 4], "pdf": [1, 2, 3, 4, 7, 9, 10], "Then": [1, 2, 3, 4, 7, 8, 9], "jupyt": [1, 2, 3, 4, 7, 10], "nbconvert": [1, 2, 3, 4], "yourlastnam": 1, "_hw1": [1, 2], "work": [1, 2, 3, 4, 5, 7, 9], "renam": 1, "file": [1, 2, 3, 4], "possibl": 1, "caus": [1, 9], "error": [1, 3, 9], "open": [1, 3], "colab": [1, 3, 4], "button": [1, 3], "just": [1, 2, 7, 9], "delet": 1, "go": 1, "cell": [1, 4], "latex": 1, "aren": 1, "visibl": 1, "search": 1, "comment": 1, "half": [1, 3], "until": 1, "bug": [1, 3], "hw": 1, "isn": 1, "meant": 1, "burden": 1, "quick": 1, "easi": [1, 9], "approach": [1, 2, 3], "save": [1, 2, 4, 10], "cut": [1, 2, 3], "off": [1, 2], "grade": [1, 5], "them": [1, 2, 4, 10], "post": [1, 7], "ed": 1, "oh": 1, "submit": 1, "instal": [1, 2, 3, 4, 10], "anaconda": [1, 2, 3, 4], "packag": [1, 2, 3, 4, 7, 8, 10], "manag": [1, 2, 3, 4], "conda": [1, 2, 3, 4], "c": [1, 2, 3, 4], "upload": [1, 2, 3, 4], "gradescop": [1, 2, 3, 4], "tag": 1, "investig": [2, 3, 4], "bayesian": [2, 3, 5], "perform": [2, 9, 10], "infer": [2, 3, 5, 7, 9], "hybrid": 2, "2004": 2, "elect": 2, "georg": 2, "bush": 2, "john": 2, "kerri": 2, "close": [2, 3, 4, 8, 10], "contest": 2, "campaign": 2, "focus": 2, "heavili": 2, "swing": 2, "state": [2, 5], "ohio": 2, "worth": 2, "vote": 2, "elector": 2, "colleg": [2, 9], "cnn": 2, "usa": 2, "todai": 2, "gallup": 2, "conduct": 2, "sever": 2, "month": 2, "lead": 2, "novemb": 2, "4th": 2, "obtain": [2, 4, 8, 9], "result": [2, 4, 9], "sept": 2, "284": 2, "344": 2, "628": 2, "28": [2, 3, 5, 9, 10], "312": 2, "325": 2, "637": [2, 10], "oct": 2, "346": 2, "339": 2, "685": 2, "31": [2, 10], "556": 2, "511": 2, "1067": 2, "although": 2, "dead": 2, "link": [2, 5, 10], "page": [2, 4], "x_i": 2, "n_i": 2, "survei": 2, "repres": [2, 4], "quantiti": [2, 9], "integ": [2, 3], "pytorch": [2, 3, 5, 7, 8, 10], "distribut": [2, 5, 10], "binomi": 2, "manual_se": [2, 3, 4, 7, 8, 9, 10], "305": [2, 3, 4, 7, 8, 9, 10], "cm": [2, 3, 7, 9, 10], "blue": [2, 3, 4, 7, 9, 10], "xs": [2, 8, 10], "ns": 2, "reason": [2, 9], "voter": 2, "bernoulli": [2, 3], "specif": [2, 3, 4, 7, 9, 10], "believ": [2, 9], "differ": [2, 3, 9], "sinc": [2, 3, 4, 7, 9, 10], "reach": [2, 3], "subpopul": 2, "g": [2, 4], "telephon": 2, "anoth": [2, 8], "internet": 2, "howev": [2, 3, 9], "highli": [2, 4], "correl": 2, "due": [2, 5], "level": [2, 5], "popul": 2, "therefor": 2, "sim": [2, 3, 4, 7, 8, 9, 10], "quad": [2, 3, 7], "bin": [2, 7, 9, 10], "bishop": [2, 5], "ch": [2, 5, 9], "unfortun": 2, "instead": [2, 3, 7, 9], "assum": [2, 3, 4, 7, 8], "ga": [2, 7], "c_": 2, "d_": 2, "fix": [2, 3, 4, 7], "parametr": 2, "interpret": 2, "recal": [2, 4, 10], "pseudo": 2, "view": 2, "previous": [2, 3], "deviat": [2, 3, 4, 9, 10], "measur": [2, 7, 9], "previou": 2, "sqrt": [2, 3, 4, 7, 8, 9, 10], "similarli": [2, 3], "adjust": 2, "belief": 2, "order": [2, 3], "must": [2, 4], "abl": [2, 3], "unobserv": 2, "rho_1": 2, "dot": [2, 4], "rho_4": 2, "simultan": 2, "block": [2, 3, 10], "first": [2, 3, 4, 10], "show": [2, 4, 7, 10], "densiti": [2, 3, 7, 8, 9, 10], "proport": [2, 4], "simplifi": [2, 9], "nice": [2, 4], "gibbs_sample_rho": 2, "length": [2, 3, 4, 9], "global": [2, 10], "usual": [2, 9], "unlik": [2, 10], "earlier": 2, "express": [2, 3], "unknown": [2, 10], "constant": [2, 4, 9], "By": [2, 4], "logarithm": 2, "addit": 2, "lgamma": 2, "alpha_log_cond": 2, "c_alpha": 2, "d_alpha": 2, "log_cond": 2, "similar": [2, 9], "beta_log_cond": 2, "c_beta": 2, "d_beta": 2, "algorithm": [2, 3, 4, 5, 10], "center": [2, 3, 7, 9], "current": [2, 3, 4, 8, 10], "propos": 2, "q": [2, 4, 10], "l": 2, "requir": [2, 5, 7, 8, 10], "li": 2, "outsid": 2, "convinc": 2, "yourself": 2, "accept": 2, "step": [2, 4, 10], "rand": 2, "uniform": 2, "mh_step_alpha_beta": 2, "mh": 2, "new_alpha": 2, "new_beta": 2, "neg": [2, 4], "reject": 2, "immedi": 2, "otherwis": [2, 3], "els": [2, 7, 8, 10], "sample_rho": 2, "estim": [2, 4, 7, 9], "posterior": [2, 3, 8, 10], "latter": 2, "broader": 2, "particular": [2, 9], "interest": [2, 8], "who": [2, 3], "win": 2, "track": [2, 10], "throughout": 2, "iter": [2, 3, 4, 10], "whether": [2, 3], "correctli": 2, "easili": 2, "recommend": [2, 4], "rhos_0": 2, "n_iter": [2, 4], "025": 2, "burn_in": 2, "initi": [2, 3, 4, 10], "fraction": [2, 3], "discard": [2, 10], "befor": [2, 9], "rhos_mean": 2, "rhos_std": 2, "prob_mean": 2, "prob_std": 2, "lp": [2, 3, 4, 10], "size": [2, 3, 4, 7, 10], "rhos_sampl": 2, "prob_sampl": 2, "rang": [2, 3, 4, 7, 8, 9, 10], "resampl": [2, 3], "singl": 2, "stack": [2, 3, 10], "axi": [2, 3, 4, 7, 8, 9, 10], "std": [2, 9, 10], "10000": [2, 3, 7, 9, 10], "second": [2, 4], "default": [2, 4], "signatur": 2, "evolut": 2, "seem": [2, 3], "favor": 2, "describ": 2, "slow": 2, "mix": [2, 4, 5, 7], "why": [2, 4], "might": [2, 3, 4], "would": [2, 8, 9], "better": [2, 3], "wai": [2, 3, 8], "markov": [2, 5], "chain": [2, 5], "transit": 2, "could": [2, 3, 4, 8, 9], "tune": 2, "converg": [2, 3], "yournam": [2, 3], "back": [2, 9], "option": 2, "browser": 2, "figur": [2, 3, 4], "explor": 3, "synthet": [3, 4], "digit": 3, "artifici": 3, "pixel": [3, 4], "well": [3, 5, 7, 10], "reconstruct": 3, "imag": 3, "applic": 3, "bit": [3, 9], "contriv": 3, "real": 3, "world": 3, "markowitz": 3, "et": [3, 5, 9, 10], "al": [3, 5, 9, 10], "2018": 3, "techniqu": [3, 8], "low": 3, "embed": 3, "partial": 3, "occlud": 3, "mice": 3, "along": 3, "build": [3, 4, 9], "intuit": 3, "hone": 3, "skill": 3, "multivari": [3, 5], "matric": [3, 4], "call": 3, "torchvis": 3, "tqdm": [3, 4, 10], "auto": [3, 4, 10], "trang": [3, 4, 10], "train": 3, "float32": 3, "subset": 3, "x3d_true": 3, "root": [3, 4, 8], "none": [3, 4, 7, 8, 9, 10], "type": [3, 7, 8], "strictli": 3, "weird": 3, "numer": [3, 9], "3": [3, 5, 7, 8, 9, 10], "three": [3, 4], "circl": 3, "radiu": 3, "speckl": 3, "random_line_mask": 3, "num_sampl": 3, "mask_siz": 3, "lw": 3, "orient": 3, "norm": 3, "dim": 3, "keepdim": 3, "xy": 3, "coordin": [3, 5, 8], "xp": 3, "yph": 3, "yp": 3, "project": 3, "onto": 3, "meshgrid": [3, 7, 8], "column_stack": [3, 8], "ravel": 3, "xpyp": 3, "num_point": 3, "unsqueez": 3, "threshold": 3, "random_circle_mask": 3, "std_origin": 3, "mean_radiu": 3, "df_radiu": 3, "circular": 3, "origin": [3, 4], "radii": 3, "determin": [3, 4, 9], "insid": 3, "correspond": [3, 4], "random_speckle_mask": 3, "p_miss": 3, "piss": 3, "p_speckl": 3, "booltensor": 3, "line_mask": 3, "circ_mask": 3, "spck_mask": 3, "len": [3, 7, 10], "mask3d": 3, "cat": 3, "randperm": 3, "substitut": 3, "255": 3, "max": [3, 9], "uint8": 3, "x3d": 3, "clone": 3, "few": [3, 4, 9, 10], "fig": [3, 4, 8, 9, 10], "ax": [3, 4, 8, 9, 10], "subplot": [3, 4, 8, 9, 10], "5": [3, 4, 5, 7, 8, 9, 10], "figsiz": [3, 4, 8, 9, 10], "imshow": [3, 4], "interpol": 3, "set_xtick": 3, "set_ytick": 3, "suptitl": 3, "store": [3, 4], "60000": 3, "784": 3, "consid": [3, 9], "x_true": 3, "reserv": 3, "valid": 3, "happen": 3, "rescal": [3, 10], "explain": [3, 4], "pc": 3, "whose": 3, "full": 3, "orthogon": [3, 8], "var_explain": 3, "plot_pca": 3, "helper": 3, "scree": 3, "set_titl": [3, 4, 9, 10], "tight_layout": [3, 7, 9, 10], "cumsum": 3, "xlim": [3, 7, 8], "ylim": [3, 7, 8, 9], "compar": 3, "far": 3, "fewer": 3, "90": [3, 10], "treat": 3, "sigma_d": 3, "tfrac": [3, 7, 9], "kappa_0": [3, 4, 7, 8, 9, 10], "mu_d": 3, "ad": 3, "z": [3, 4, 8, 10], "diag": [3, 4, 8], "mu_1": 3, "sigma_1": [3, 10], "graphic": [3, 5, 9], "omit": 3, "th": [3, 7, 9, 10], "On": 3, "formal": 3, "mathsf": [3, 7], "ob": 3, "respect": [3, 4, 9], "goal": [3, 7], "hyperparamt": 3, "altern": [3, 9], "With": 3, "approxim": [3, 10], "region": 3, "w_d": 3, "_i": 3, "neq": 3, "mu_i": 3, "sigma_i": 3, "upon": 3, "udpat": 3, "finish": 3, "header": 3, "organ": [3, 9], "solut": [3, 4], "sigmasq": [3, 7, 10], "nu0": [3, 4, 7, 8], "sigmasq0": [3, 7], "kappa0": [3, 4, 7], "lambda0": 3, "_complete_": 3, "fill": [3, 4], "advantag": 3, "broadcast": [3, 7, 9], "gibbs_sample_lat": 3, "capabl": 3, "give": [3, 4], "memori": 3, "issu": 3, "crash": 3, "kernel": 3, "gibbs_sample_weight": 3, "gibbs_sample_mean": 3, "gibbs_sample_vari": 3, "gibbs_sample_missing_data": 3, "boolean": 3, "sort": 3, "index": [3, 7, 8], "org": [3, 4], "cppdoc": 3, "tensor_index": 3, "html": [3, 4, 10], "val": 3, "1d": [3, 9], "200": [3, 4, 9], "minut": 3, "my": 3, "01": [3, 10], "n_sampl": [3, 9, 10], "dictionari": [3, 10], "tausq": [3, 9, 10], "theta": [3, 9, 10], "overwrit": 3, "pass": [3, 7, 8], "high": [3, 9], "fmask": 3, "n_ob": 3, "x_mean": 3, "randomli": 3, "output": [3, 4, 10], "itr": [3, 10], "cycl": [3, 10], "append": [3, 4, 10], "combin": [3, 4, 10], "cool": [3, 10], "zip": [3, 8, 9, 10], "samples_dict": [3, 10], "kei": [3, 10], "x_miss": 3, "6": [3, 4, 5, 7, 8, 9, 10], "min": 3, "debug": 3, "reduc": 3, "reset": 3, "final": 3, "trace": [3, 10], "averag": [3, 9, 10], "last": [3, 4, 9, 10], "arrang": 3, "5x5": 3, "shown": 3, "28x28": 3, "squar": [3, 4, 7, 8], "offset": 3, "x_recon": 3, "im": [3, 4], "vmin": 3, "vmax": 3, "across": 3, "rmse": 3, "xtick": [3, 9], "ytick": 3, "titl": [3, 7], "colorbar": [3, 7], "per": [3, 10], "surpris": 3, "poorli": 3, "imagin": 3, "put": [3, 9], "strang": 3, "realli": 3, "among": 3, "_r": 3, "succeq": 3, "_c": 3, "obei": 3, "special": [3, 4, 7, 9], "kroneck": 3, "mn": 3, "iff": 3, "vec": 3, "otim": 3, "cdot": 3, "oper": [3, 9], "major": 3, "suppos": [3, 7, 9], "bmatrix": 3, "4": [3, 4, 5, 7, 8, 9, 10], "left": [3, 4, 7, 8, 9, 10], "right": [3, 4, 7, 8, 9, 10], "concaten": 3, "illustr": 3, "ident": [3, 4, 8], "mind": 3, "case": [3, 7], "pi": [3, 4, 7, 8], "dm": 3, "exp": [3, 7, 8, 9], "tr": [3, 4, 8], "propto": [3, 7, 8, 9, 10], "appropri": 3, "contrast": 3, "wikipedia": [3, 4], "ther": 3, "flip": 3, "familiar": 3, "remov": 3, "_hw3": 3, "browswer": 3, "separ": [4, 9], "foreground": 4, "achiev": 4, "refer": 4, "en": [4, 10], "wiki": 4, "image_segment": 4, "cluster": 4, "k": [4, 7, 8, 9, 10], "likelihood": [4, 7, 8, 9], "mu_k": 4, "sigma_k": 4, "mu_": [4, 7], "sigma_": [4, 9], "categor": [4, 9, 10], "multinomi": 4, "color": [4, 7, 8, 9], "channel": 4, "red": 4, "green": 4, "wishart": 4, "iw": 4, "symmetr": 4, "dirichlet": [4, 5], "1_k": 4, "simplic": 4, "wish": 4, "via": [4, 10], "underset": 4, "operatornam": 4, "argmax": 4, "correct": 4, "doubl": 4, "_q": 4, "underbrac": 4, "limits_": 4, "omega_": 4, "nk": 4, "qquad": [4, 9], "sum_": [4, 7, 8, 9, 10], "pi_k": 4, "alpha_k": 4, "eta_k": 4, "phi": 4, "nu": [4, 7], "contant": 4, "explicitli": 4, "inner": 4, "langl": 4, "rangl": 4, "a_1": 4, "b_1": 4, "a_2": 4, "b_2": 4, "a_3": 4, "b_3": 4, "a_4": 4, "b_4": 4, "deduc": 4, "written": 4, "phi_": 4, "nu_": 4, "conclud": 4, "summand": 4, "niw": 4, "mode": 4, "starter": 4, "own": [4, 7, 9, 10], "entail": 4, "prove": 4, "reli": 4, "extern": 4, "those": [4, 10], "offer": 4, "tensorflow": [4, 9], "scikit": 4, "patch": 4, "ellips": [4, 8], "invwishart_log_prob": 4, "sigma0": 4, "assert": [4, 10], "logdet": 4, "diagon": 4, "dim1": 4, "dim2": 4, "multigammaln": 4, "mu0": [4, 7], "discret": 4, "logsumexp": [4, 9], "latent": [4, 10], "respons": 4, "non": [4, 7], "desir": [4, 9, 10], "alpha0": 4, "Their": 4, "consist": 4, "param": 4, "logit": 4, "prob": [4, 9], "tighten": 4, "bound": 4, "ground": 4, "truth": 4, "whichev": 4, "choos": [4, 8], "roughli": 4, "recov": 4, "displai": 4, "confidence_ellips": 4, "cov": 4, "n_std": 4, "facecolor": 4, "kwarg": [4, 7, 8], "modifi": [4, 10], "galleri": 4, "statist": [4, 5, 9], "confid": 4, "radius": 4, "forward": 4, "2d": 4, "pearson": 4, "ell_radius_x": 4, "ell_radius_i": 4, "height": 4, "multipli": 4, "rotat": 4, "translat": 4, "transf": 4, "affine2d": 4, "rotate_deg": 4, "45": [4, 10], "set_transform": 4, "transdata": 4, "add_patch": 4, "test_toi": 4, "seed": 4, "ord": 4, "n_test": 4, "300": 4, "em_result": 4, "scatter": 4, "marker": 4, "edgecolor": [4, 9, 10], "linestyl": 4, "part": 4, "coupl": [4, 8], "discuss": [4, 8], "readi": 4, "github": 4, "fox": 4, "png": 4, "cow": 4, "owl": 4, "zebra": 4, "summari": 4, "load_imag": 4, "filenam": 4, "imread": 4, "astyp": 4, "rgb": 4, "save_segment": 4, "np": 4, "set_axis_off": 4, "nan": 4, "compon": [4, 5], "savefig": [4, 9, 10], "run_segment": 4, "gmm": 4, "_seg": 4, "21": [4, 5, 7, 10], "best": 4, "extend": 4, "inform": 4, "hw4_yournam": 4, "instructor": 5, "scott": 5, "linderman": 5, "ta": 5, "xavier": 5, "gonzalez": 5, "probabilist": [5, 9, 10], "topic": 5, "includ": 5, "mcmc": 5, "variat": [5, 7, 8], "reduct": 5, "princip": 5, "factor": 5, "extens": 5, "involv": 5, "program": [5, 9, 10], "comfort": 5, "calculu": 5, "linear": [5, 9], "algebra": 5, "emphas": 5, "profici": 5, "tuesdai": 5, "thursdai": 5, "30": [5, 9, 10], "50am": 5, "advanc": 5, "undergrad": 5, "basi": [5, 8], "credit": 5, "letter": 5, "offic": 5, "wed": 5, "30pm": 5, "wu": 5, "tsai": 5, "neurosci": 5, "instiut": 5, "room": 5, "m252g": 5, "thur": 5, "7pm": 5, "locat": 5, "s275": 5, "releas": [5, 7, 8, 10], "fridai": 5, "59pm": 5, "primarili": 5, "murphi": 5, "machin": 5, "mit": 5, "press": 5, "pattern": 5, "recognit": 5, "york": 5, "springer": 5, "2006": 5, "gelman": [5, 9, 10], "chapman": 5, "hall": 5, "2005": 5, "date": 5, "apr": 5, "slide": 5, "2": [5, 7, 8, 9], "mont": 5, "carlo": 5, "1": [5, 7, 8, 9, 10], "pca": 5, "hamiltonian": 5, "neal": 5, "2012": 5, "27": [5, 10], "maxim": 5, "membership": 5, "ascent": 5, "blei": 5, "2017": 5, "autoencod": 5, "kingma": 5, "2019": [5, 9], "amort": 5, "hidden": 5, "29": [5, 10], "dynam": 5, "system": 5, "process": 5, "stochast": 5, "equat": 5, "june": 5, "wrap": [5, 10], "studentt": 7, "talk": 7, "sat": [7, 9], "score": [7, 10], "sample_shap": [7, 8], "lkhd": 7, "linspac": [7, 8, 9, 10], "500": [7, 9], "label": [7, 8, 9, 10], "05": [7, 9, 10], "enumer": [7, 8], "03": [7, 10], "legend": [7, 8, 9, 10], "fontsiz": [7, 9, 10], "0x7f96cc3fa1f0": 7, "j_n": 7, "h_n": 7, "tild": 7, "ml": 7, "maximum": 7, "4590": 7, "4819": 7, "purpl": 7, "0x7f97a00fde50": 7, "calcul": 7, "littl": [7, 9, 10], "simpler": 7, "reparameter": 7, "equiv": 7, "scaledchisq": 7, "sample_sum_sq": 7, "present": 7, "zs": [7, 8, 10], "match": 7, "empir": 7, "chisq": 7, "lmbda": [7, 8], "ec": [7, 10], "0x7f96ca1a1dc0": 7, "vari": [7, 9], "dens": [7, 9], "1e": [7, 9], "Its": 7, "chang": [7, 9], "formula": 7, "f": [7, 9], "iga": 7, "again": [7, 9], "parameter": 7, "inv_chisq": 7, "0x7f96c9124af0": 7, "triangleq": [7, 9], "mu_grid": [7, 9], "sigmasq_grid": 7, "contourf": 7, "sample_nix": 7, "markeredgecolor": 7, "markerfacecolor": 7, "0f": [7, 9], "opt": [7, 8, 10], "hostedtoolcach": [7, 8, 10], "x64": [7, 8, 10], "lib": [7, 8, 10], "python3": [7, 8, 10], "site": [7, 8, 10], "py": [7, 8, 10], "504": [7, 8], "userwarn": [7, 8], "upcom": [7, 8], "argument": [7, 8], "trigger": [7, 8], "intern": [7, 8], "aten": [7, 8], "src": [7, 8], "nativ": [7, 8], "tensorshap": [7, 8], "cpp": [7, 8], "3483": [7, 8], "_vf": [7, 8], "ignor": [7, 8], "attr": [7, 8], "st": 7, "kappa": 7, "mu_margin": 7, "cover": 7, "bunch": 7, "math": 7, "z_1": 8, "z_d": 8, "bf": 8, "contour": 8, "z1": 8, "z2": 8, "logpdf": 8, "z_2": 8, "gca": 8, "set_aspect": 8, "u": 8, "lambda_1": 8, "lambda_d": 8, "linearli": 8, "theta1": [8, 10], "u1": 8, "co": 8, "sin": 8, "theta2": 8, "u2": 8, "eigenval": 8, "lmbda1": 8, "lmbda2": 8, "pick": [8, 9], "isocontour": 8, "arrow": 8, "head_width": 8, "u_1": 8, "lambda_2": 8, "u_2": 8, "plot_cov": 8, "eigendecomposit": 8, "eigh": 8, "aspect": 8, "lmbda0": 8, "covariance_matrix": 8, "sharex": [8, 9, 10], "sharei": 8, "set_xlim": [8, 10], "set_ylim": [8, 10], "set_ylabel": [8, 9, 10], "set_xlabel": [8, 9, 10], "253": 8, "singular": 8, "detect": 8, "warn": 8, "recogn": 8, "yet": 8, "weak": 8, "lmbda_0": 8, "lmbda_n": 8, "precision_sampl": 8, "covariance_sampl": 8, "markers": 8, "sai": [8, 9], "nw": 8, "kappa_n": 8, "outer": 8, "posterior_lambda": 8, "lambda_sampl": 8, "sigma_sampl": 8, "posterior_mu": 8, "precision_matrix": 8, "mu_sampl": [8, 9, 10], "ro": 8, "mec": 8, "0x7f7566b14bb0": 8, "gave": 8, "theori": 8, "matrixinversetransform": 8, "particularli": 8, "hard": 8, "chose": 8, "practic": 8, "practition": 8, "suggest": 8, "lkjcholeski": 8, "technic": 8, "doesn": 8, "amen": 8, "later": 8, "These": 9, "most": 9, "complex": 9, "still": 9, "exact": 9, "motiv": 9, "n_": [9, 10], "aim": 9, "allow": 9, "studi": 9, "rel": 9, "individu": 9, "exchang": 9, "group": 9, "themselv": 9, "tau": 9, "tau_0": [9, 10], "theta_": 9, "henc": 9, "bar": [9, 10], "_s": [9, 10], "suffici": 9, "2013": [9, 10], "numpyro": 9, "educ": 9, "servic": 9, "analyz": 9, "coach": 9, "v": 9, "scholast": 9, "aptitud": 9, "verbal": 9, "administr": 9, "choic": 9, "administ": 9, "admiss": 9, "decis": 9, "800": 9, "examin": 9, "resist": 9, "short": 9, "effort": 9, "direct": 9, "toward": 9, "improv": 9, "reflect": 9, "knowledg": 9, "acquir": 9, "abil": 9, "develop": 9, "year": 9, "nevertheless": 9, "success": 9, "increas": 9, "odd": 9, "directli": 9, "treatment": 9, "control": 9, "stuff": 9, "magnitud": 9, "nix": [9, 10], "tausq_0": [9, 10], "x_bar": [9, 10], "sigma_bar": [9, 10], "doabl": 9, "good": 9, "exercis": 9, "pain": 9, "hold": 9, "v_": [9, 10], "bay": 9, "denomin": 9, "disappear": 9, "lambda_": 9, "complic": 9, "quadratur": 9, "compute_log_f": 9, "log_f": 9, "handl": 9, "v_mu": [9, 10], "mu_hat": [9, 10], "exponenti": 9, "tausq_grid": 9, "256": 9, "dt": 9, "p_tausq": 9, "401": 9, "harder": 9, "appli": 9, "tau_grid": 9, "p_tau": 9, "tau_mean": 9, "diff": 9, "tau_var": 9, "3f": 9, "459": 9, "541": 9, "intract": 9, "analyt": 9, "compute_posterior_mu": 9, "var": 9, "p_mu": 9, "big": 9, "compute_posterior_theta": 9, "v_theta": [9, 10], "theta_hat": [9, 10], "least": 9, "_2": 9, "affect": 9, "compute_posterior_theta2": 9, "syntax": 9, "elementwis": 9, "pair": 9, "p_theta": 9, "ln": 9, "fill_between": 9, "get_color": 9, "ind": 9, "tau_sampl": [9, 10], "theta_sampl": 9, "median": 9, "quantil": 9, "95": [9, 10], "6455": 9, "8015": 9, "7383": 9, "7419": 9, "0388": 9, "7982": 9, "5835": 9, "9828": 9, "3350": 9, "2647": 9, "6378": 9, "3968": 9, "1119": 9, "3157": 9, "7577": 9, "8330": 9, "7211": 9, "5044": 9, "7342": 9, "4147": 9, "6808": 9, "8380": 9, "5527": 9, "0518": [9, 10], "4878": 9, "8416": 9, "1996": 9, "0148": 9, "0938": 9, "3629": 9, "0491": 9, "6214": 9, "1064": 9, "1661": 9, "5029": 9, "3792": 9, "8829": 9, "5121": 9, "2860": 9, "str": [9, 10], "even": 9, "though": 9, "had": 9, "sensit": 9, "util": 10, "pyro": 10, "languag": 10, "pip": 10, "ppl": 10, "1m": 10, "0m": 10, "34": 10, "49mnotic": 10, "39": 10, "49m": 10, "avail": 10, "49m22": 10, "32": 10, "49m23": 10, "49mpip": 10, "upgrad": 10, "school": 10, "alpha_0": 10, "op": 10, "effective_sample_s": 10, "autocorrel": 10, "tqdmwarn": 10, "iprogress": 10, "ipywidget": 10, "readthedoc": 10, "io": 10, "stabl": 10, "user_instal": 10, "autonotebook": 10, "notebook_tqdm": 10, "shcool": 10, "effect": 10, "allclos": 10, "idea": 10, "remain": 10, "often": 10, "deriv": 10, "sampler": 10, "progress": 10, "parallel": 10, "gibbs_sample_theta": 10, "everyth": 10, "scala": 10, "drawn": 10, "alpha_n": 10, "gibbs_sample_sigmasq": 10, "gibbs_sample_mu": 10, "tau_n": 10, "gibbs_sample_tausq": 10, "tausq_n": 10, "9999": 10, "00": 10, "65": 10, "645": 10, "65it": 10, "130": 10, "643": 10, "54it": 10, "195": 10, "642": 10, "260": 10, "639": 10, "75it": 10, "324": 10, "61it": 10, "389": 10, "93it": 10, "454": 10, "12it": 10, "519": 10, "644": 10, "01it": 10, "584": 10, "69it": 10, "649": 10, "636": 10, "713": 10, "632": 10, "40it": 10, "777": 10, "631": 10, "49it": 10, "841": 10, "23it": 10, "906": 10, "634": 10, "24it": 10, "971": 10, "638": 10, "15it": 10, "1036": 10, "640": 10, "56it": 10, "1101": 10, "50it": 10, "1167": 10, "1233": 10, "647": 10, "16it": 10, "1299": 10, "02": 10, "43it": 10, "1364": 10, "648": 10, "89it": 10, "1429": 10, "646": 10, "81it": 10, "1495": 10, "13it": 10, "1561": 10, "652": 10, "29it": 10, "1627": 10, "653": 10, "05it": 10, "1693": 10, "82it": 10, "1759": 10, "08it": 10, "1825": 10, "654": 10, "74it": 10, "1891": 10, "95it": 10, "1957": 10, "651": 10, "47it": 10, "39it": 10, "2088": 10, "625": 10, "2152": 10, "629": 10, "2217": 10, "2282": 10, "96it": 10, "2347": 10, "77it": 10, "2413": 10, "2478": 10, "21it": 10, "2543": 10, "26": 10, "2609": 10, "04": 10, "26it": 10, "2674": 10, "2739": 10, "633": 10, "91it": 10, "2803": 10, "630": 10, "2867": 10, "19it": 10, "2930": 10, "439": 10, "31it": 10, "2995": 10, "485": 10, "97it": 10, "3060": 10, "524": 10, "3124": 10, "553": 10, "3188": 10, "576": 10, "09it": 10, "3249": 10, "573": 10, "33": 10, "3311": 10, "585": 10, "06it": 10, "3375": 10, "599": 10, "33it": 10, "3440": 10, "611": 10, "35": 10, "3505": 10, "620": 10, "59it": 10, "36": 10, "3570": 10, "626": 10, "3635": 10, "04it": 10, "37": 10, "3699": 10, "09": 10, "51it": 10, "38": 10, "3763": 10, "06": 10, "17it": 10, "3827": 10, "60it": 10, "3892": 10, "79it": 10, "40": 10, "3957": 10, "66it": 10, "4022": 10, "641": 10, "62it": 10, "41": 10, "4087": 10, "34it": 10, "42": 10, "4152": 10, "68it": 10, "4217": 10, "55it": 10, "43": 10, "4282": 10, "08": 10, "4347": 10, "72it": 10, "44": 10, "4412": 10, "07": 10, "4477": 10, "4542": 10, "46": 10, "4607": 10, "47": 10, "4673": 10, "4738": 10, "32it": 10, "48": 10, "4803": 10, "49": 10, "4868": 10, "44it": 10, "4933": 10, "4998": 10, "14it": 10, "51": 10, "5063": 10, "98it": 10, "5128": 10, "90it": 10, "52": 10, "5193": 10, "92it": 10, "53": 10, "5258": 10, "5323": 10, "54": 10, "5388": 10, "38it": 10, "55": 10, "5453": 10, "5518": 10, "46it": 10, "56": 10, "5584": 10, "5649": 10, "57": 10, "5714": 10, "58": 10, "5780": 10, "5845": 10, "45it": 10, "59": 10, "5910": 10, "60": 10, "5976": 10, "87it": 10, "6042": 10, "35it": 10, "61": 10, "6108": 10, "62": 10, "6173": 10, "6239": 10, "650": 10, "52it": 10, "63": 10, "6305": 10, "64": 10, "6370": 10, "6436": 10, "6502": 10, "66": 10, "6568": 10, "37it": 10, "6634": 10, "67": 10, "6699": 10, "68": 10, "6764": 10, "6829": 10, "69": 10, "6894": 10, "20it": 10, "70": 10, "6960": 10, "7025": 10, "71": 10, "7091": 10, "72": 10, "7156": 10, "71it": 10, "7221": 10, "73": 10, "7287": 10, "74": 10, "7352": 10, "7417": 10, "27it": 10, "75": 10, "7483": 10, "7548": 10, "76": 10, "7614": 10, "02it": 10, "77": 10, "7679": 10, "67it": 10, "7743": 10, "78": 10, "7807": 10, "627": 10, "79": 10, "7871": 10, "7935": 10, "7999": 10, "81": 10, "8063": 10, "8129": 10, "635": 10, "80it": 10, "82": 10, "8195": 10, "70it": 10, "83": 10, "8261": 10, "8327": 10, "84": 10, "8393": 10, "85": 10, "8459": 10, "8524": 10, "86": 10, "8589": 10, "00it": 10, "87": 10, "8654": 10, "8720": 10, "88": 10, "8786": 10, "63it": 10, "89": 10, "8852": 10, "07it": 10, "8918": 10, "11it": 10, "8984": 10, "91": 10, "9050": 10, "9116": 10, "85it": 10, "92": 10, "9182": 10, "94it": 10, "9247": 10, "93": 10, "9312": 10, "94": 10, "9377": 10, "9442": 10, "9507": 10, "22it": 10, "96": 10, "9572": 10, "30it": 10, "9637": 10, "53it": 10, "97": 10, "9702": 10, "98": 10, "9767": 10, "9832": 10, "99": 10, "9898": 10, "9963": 10, "25it": 10, "lps1": 10, "simpli": 10, "burn": 10, "subsequ": 10, "analys": 10, "burnin": 10, "lps2": 10, "onward": 10, "theta_samples1": 10, "theta_samples2": 10, "sigma_samples1": 10, "sigma_samples2": 10, "acf_tausq": 10, "acf_mu": 10, "acf_theta1": 10, "acf_sigma1": 10, "250": 10, "theta_1": 10, "lag": 10, "acf": 10, "sigamsq1": 10, "1563": 10, "3730": 10, "2454": 10, "1027": 10, "9816": 10, "8996": 10, "complt": 10, "rest": 10, "monitor": 10}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"hw0": 0, "pytorch": [0, 1], "primer": 0, "1": [0, 1, 2, 3, 4], "construct": 0, "tensor": [0, 3], "problem": [0, 1, 2, 3, 4], "2": [0, 1, 2, 3, 4, 10], "3": [0, 1, 2, 4], "4": [0, 1, 2], "5": [0, 1, 2], "6": [0, 1, 2], "broadcast": 0, "fanci": 0, "index": 0, "distribut": [0, 1, 3, 4, 7, 8, 9], "hw1": 1, "bayesian": [1, 4, 7, 8, 10], "linear": 1, "regress": 1, "deriv": [1, 2, 3, 4], "posterior": [1, 4, 7, 9], "math": [1, 2, 3, 4], "The": [1, 3, 8], "mean": [1, 7, 8, 9], "synthet": [1, 10], "data": [1, 2, 3, 7, 10], "comput": [1, 7, 9], "code": [1, 2, 3, 4, 7], "plot": [1, 3, 7, 8, 10], "densiti": 1, "varianc": [1, 7, 9], "sampl": [1, 2, 3, 8, 9, 10], "function": [1, 3, 4, 8], "predict": 1, "6a": 1, "6b": 1, "submiss": [1, 2, 3, 4], "instruct": [1, 2, 3, 4], "hw2": 2, "gibb": [2, 3, 10], "metropoli": 2, "hast": 2, "background": [2, 4], "poll": 2, "hierarch": [2, 9, 10], "model": [2, 3, 4, 7, 9, 10], "complet": [2, 3], "condit": [2, 3], "rho": 2, "part": [2, 3], "demonstr": 2, "independ": 2, "rho_i": 2, "b": 2, "write": [2, 3, 7, 8], "from": [2, 8, 9], "implement": [2, 3, 4], "alpha": 2, "evalu": 2, "unnorm": 2, "log": [2, 4, 10], "probabl": [2, 4, 10], "beta": 2, "sampler": [2, 3], "7": 2, "diagnost": [2, 10], "8": 2, "reflect": 2, "hw3": 3, "continu": 3, "latent": 3, "variabl": 3, "setup": [3, 4, 10], "download": 3, "mnist": 3, "dataset": [3, 4], "simpl": 3, "mask": 3, "off": 3, "some": 3, "make": 3, "appli": 3, "them": 3, "each": [3, 4], "point": 3, "flatten": 3, "2d": 3, "princip": 3, "compon": 3, "analysi": [3, 7], "svd": 3, "1a": [3, 4], "run": [3, 4], "pca": 3, "directli": 3, "1b": [3, 4], "short": [3, 4], "answer": [3, 4], "why": 3, "doe": 3, "need": 3, "so": 3, "mani": [3, 8], "more": 3, "factor": 3, "miss": 3, "2a": [3, 4], "2b": [3, 4], "which": 3, "step": 3, "can": 3, "perform": [3, 4], "parallel": 3, "2c": [3, 4], "provid": 3, "your": [3, 4], "result": 3, "discuss": 3, "bonu": 3, "matrix": 3, "normal": [3, 7, 8], "weight": 3, "under": 3, "prior": 3, "hw4": 4, "mixtur": 4, "em": 4, "calcul": [4, 10], "q_n": 4, "z_n": 4, "p": 4, "x_n": 4, "theta": 4, "expect": 4, "1c": 4, "expand": 4, "mathcal": 4, "l": 4, "_1": 4, "exponenti": 4, "famili": 4, "form": 4, "1d": 4, "maxim": 4, "1e": 4, "_2": 4, "gaussian": [4, 9, 10], "helper": 4, "log_prob": 4, "e_step": 4, "m_step": 4, "given": [4, 7, 9], "test": [4, 9], "toi": 4, "imag": 4, "segment": 4, "final": [4, 7], "3a": 4, "multipl": 4, "restart": 4, "3b": 4, "improv": 4, "overview": 5, "cours": 5, "descript": 5, "prerequisit": 5, "logist": 5, "book": 5, "schedul": 5, "refer": 6, "unknown": [7, 8], "warm": 7, "up": 7, "precis": [7, 8], "exercis": 7, "nix": 7, "over": [7, 9], "margin": [7, 9], "recap": [7, 8, 9, 10], "multivari": 8, "gener": 8, "stori": 8, "visual": 8, "covari": 8, "matric": 8, "draw": [8, 9], "wishart": 8, "invers": 8, "infer": [8, 10], "score": 9, "across": 9, "school": 9, "eight": 9, "exampl": 9, "global": 9, "per": 9, "effect": 9, "question": 9, "markov": 10, "chain": 10, "mont": 10, "carlo": 10, "mcmc": 10, "joint": 10, "updat": 10, "theta_": 10, "sigma_": 10, "mu": 10, "tau": 10, "put": 10, "all": 10, "togeth": 10, "summari": 10}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})