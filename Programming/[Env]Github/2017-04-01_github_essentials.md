# Github Essentials  
\# 2017-04-01 written by jkang  
\# ref: <http://nvie.com/posts/a-successful-git-branching-model/>  

This tutorial assumes you had ```dev``` branch on which your major works are done, and you want to merge ```dev``` to ```master``` branch  

Make sure set up an editor on Terminal as your preference. I prefer to working on Emacs, but Github uses vim as defaul editor, so change the default editor use the following command:  
```>> git config --global core.editor "emacs"```  
For Sublime, use  
```>> git config --global core.editor "subl"```  

### (1) After making changes on local repo (```dev```), type in Terminal as:

```>> git branch```  # check your branch first as  ```dev``` not ```master```  
```>> git status```  # see your changes  
```>> git add .```  
```>> git status```  # see your updates  
```>> git commit -m 'update'```  
```>> git push```  

### (2) To merge ```dev``` to ```master```, type in Terminal as:  
Make sure you are pointing to ```master``` branch first  
```>> git checkout master```  
```>> git merge --no-ff dev```  
When the text editor appears, add comments about the changes and then save it. ```--no-ff```option allows you to record change histories on Github (recommended)  
```>> git push origin master```  # finally update the changes to ```master```  
```>> git checkout dev```  # make sure you change back to ```dev``` branch!!   
