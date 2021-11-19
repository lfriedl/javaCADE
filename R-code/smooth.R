bn.smooth = function(net, epsilon = 1e-15){
    new.dist = sapply(names(net), function(name){
        #remember the old dimensions and variable names
        old.CPD = net[[name]]$prob
        old.dim = dim(old.CPD)
        old.dimnames = dimnames(old.CPD)

        #now flatten the table into a vector and smooth
        CPD.mat = as.vector(old.CPD)

        var.size = length(old.dimnames[[name]])
        if(var.size == 0){
            print(CPD.mat)
            var.size = length(CPD.mat)
        }
	    for(i in seq(1, length(CPD.mat), var.size) ){
            CPD.mat[i:(i + var.size -1)] = CPD.mat[i:(i + var.size -1)] + epsilon
            CPD.mat[i:(i + var.size -1)] = CPD.mat[i:(i + var.size -1)] / sum(CPD.mat[i:(i + var.size -1)])
        }

	    #set the NAs to epsilon
        CPD.mat[is.na(CPD.mat)] = 1/var.size

        #set the dimension and variable names back, cast to table and return
        dim(CPD.mat) = dim(old.CPD)
        dimnames(CPD.mat) = old.dimnames
        return(as.table(CPD.mat))
    }, simplify=FALSE, USE.NAMES=TRUE)
    print(new.dist)
    return(custom.fit(bn.net(net), dist = new.dist))
}
