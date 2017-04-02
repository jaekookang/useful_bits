# Github Essentials  
2017-04-01 written by jkang

This tutorial assumes you had ```dev``` branch on which your major works are done, and you want to merge ```dev``` to ```master``` branch  

### (1) After making changes on local repo (```dev```), type in Terminal as:
```>> git branch```  # check your branch first as ```dev``` not ```master```  
```>> git status```  # see your changes  
```>> git add .```  
```>> git status```  # see your updates
```>> git commit -m 'update'```  
```>> git push```  

### (2) To merge ```dev``` to ```master```, type in Terminal as:  
Make sure you are pointing to ```master``` first  
```>> git checkout master```  
```>> git merge --no-ff dev```
```>> git```
