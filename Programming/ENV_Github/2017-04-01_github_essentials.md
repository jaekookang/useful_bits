# Github Essentials  
2017-04-01 written by jkang  
ref: <http://nvie.com/posts/a-successful-git-branching-model/>  

---

* This tutorial assumes you have a ```dev``` branch in which your major works are done, and you want to merge ```dev``` into the ```master``` branch  
* This tutorial is mainly for personal code management, but can be applied to projects by multiple users

Make sure to set up an editor in Terminal as per your preference. I prefer to work in Emacs, but Github uses vim as its default editor, so change the default editor use the following command:  
```>> git config --global core.editor "emacs"```  
For Sublime, use:  
```>> git config --global core.editor "subl"```  

### (1) After making changes on local repo (```dev```), type the following in Terminal:

```>> git branch```  # check your branch first as  ```dev``` not ```master```  
```>> git status```  # see your changes  
```>> git add .```  
```>> git status```  # see your updates  
```>> git commit -m 'update'```  
```>> git push```  

### (2) To merge ```dev``` to ```master```, type the following in Terminal:  
Make sure you are pointing to the ```master``` branch first  
```>> git checkout master```  
```>> git merge --no-ff dev```  
When the text editor appears, add comments about the changes and then save it. ```--no-ff```option allows you to record change histories on Github (recommended)  
```>> git push origin master```  # finally update the changes to ```master```  
```>> git checkout dev```  # make sure you change back to the ```dev``` branch!!   
