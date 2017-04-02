# Github Essentials  
2017-04-01 written by jkang

This tutorial assumes you had ```dev``` branch on which your major works are done, and you want to merge ```dev``` to ```master``` branch  

Make sure set up an editor on Terminal as your preference. I prefer to work on emacs, but Github uses vim as defaul editor, so change the default editor use the following command:  
```>> git config --global core.editor "emacs"```  
For Sublime, use  
```>> git config --global core.editor "subl"```  

### (1) After making changes on local repo (```dev```), type in Terminal as:

```>> git branch```  # check your branch first as  ```dev``` not ```master```  # see your changes
```>> git status```  
```>> git add .```  
```>> git status```  # see your updates  
```>> git commit -m 'update'```  
```>> git push```  

### (2) To merge ```dev``` to ```master```, type in Terminal as:  
Make sure you are pointing to ```master``` first  
```>> git checkout master```  
```>> git merge --no-ff dev```  
When the text editor appears, add comments about the changes and then save it  
```>> git push origin master```  # finally update the changes to ```master```  
```>> git checkout dev```  # make sure you change back to ```dev``` branch!!   
