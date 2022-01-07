
# log in the remote server
```shell
    ssh ky8517@tiger.princeton.edu
   
    # ssh login without password: https://www.hongkiat.com/blog/ssh-to-server-without-password/
    # create id_rsa.pub if you don't have it 
    cat id_rsa.pub | ssh ky8517@tiger.princeton.edu 'cat - >> ~/.ssh/authorized_keys'
```
# Copy remote files to local 
```commandline   
    # copy local to remote 
    tar cfz local_file |ssh user@remotehost 'cd /desired/location; tar xfz -'
    
    # copy remote to local 
    rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/results-repeats_100 ~/Downloads/
    
    ssh ky8517@tiger.princeton.edu 'cd /scratch/gpfs/ky8517/fkm/fkm/ && tar -cf - -C results-repeats_100' | tar -xf - -C ~/Downloads
    https://askubuntu.com/questions/1236768/how-to-compress-and-download-directory-from-remote-to-local
```
