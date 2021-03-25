# Scripts
All Python scripts go here. They will be sectioned off by their purpose. 
All files can be called by using main.py file.

### \_\_init\_\_.py files
If you are new to python, you may not know what  `__init__.py` files are. These files denote 
subpackages that can be used. So, for example, if I wanted to use a file under `random_forest` directory,
I could have something like

```python
import random_forrest as rf 
```

That would then allow me to call `rf.(something from random_forest directory)`. You can think of these files 
like initialization for a subpackage. You can import other packages in an `__init__.py` file, create variables
, have specific functions for that subpackage, etc. For example, if te random_forest had an `__init__.py`
like the following

```python
from xgboost import Forrest

def print_forrest(forrest):
    print("do something to print forrest here")

main_forrest = "xgboost"
```

Then you could use `rf.print_forrest(forrest)`, `rf.Forrest()`, or `print(rf.main_forrest)` using the rf
variable imported above.
