# Jupyter for Matlab
2017-04-02 jkang  
ref:  
- [Matlab kernel](https://github.com/Calysto/matlab_kernel)  
- [Matlab API](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

---

Matlab code can be visualized and presented using Jupyter  

Here is how  

First, find your matlab root directory and type as follows in Terminal:    
```>> cd matlabroot/extern/engines/python```  
```>> python setup.py install```  

```matlabroot``` is your matlab root directory  
For example, in my case  
```>> cd /Applications/MATLAB_R2016b.app/extern/engines/python```  
```>> python setup.py install```

Second, after installing Matlab API engine for Python, download _matlab\_kernel_ from pip  
```>> pip install matlab_kernel```  

This is it! Now you can use Matlab mode on Jupyter by selecting 'Matlab' under New menu  
```>> jupyter notebook ```