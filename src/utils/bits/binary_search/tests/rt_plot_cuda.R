range<-"10K-100K_1K-1M"
#device<-"GTX460"
device<-"GTX285"
#device<-"9800GT"

t_sm_name<-paste("t_sm_b_search_cuda-",range,".",device,".txt",sep="")
t_gm_name<-paste("t_gm_b_search_cuda-",range,".",device,".txt",sep="")
i_sm_name<-paste("i_sm_b_search_cuda-",range,".",device,".txt",sep="")
i_gm_name<-paste("i_gm_b_search_cuda-",range,".",device,".txt",sep="")
b_name<-paste("b_search_cuda-",range,".",device,".txt",sep="")
s_name<-paste("sort_b_search_cuda-",range,".",device,".txt",sep="")
s_t_gm_name<-paste("sort_t_gm_b_search_cuda-",range,".",device,".txt",sep="")
s_i_gm_name<-paste("sort_i_gm_b_search_cuda-",range,".",device,".txt",sep="")

t_sm <- read.delim(t_sm_name,header=F,sep=",")
t_gm <- read.delim(t_gm_name,header=F,sep=",")
i_sm <- read.delim(i_sm_name,header=F,sep=",")
i_gm <- read.delim(i_gm_name,header=F,sep=",")
b <- read.delim(b_name,header=F,sep=",")
s <- read.delim(s_name,header=F,sep=",")
s_t_gm <- read.delim(s_t_gm_name,header=F,sep=",")
s_i_gm <- read.delim(s_i_gm_name,header=F,sep=",")

d_s<-unique(b$V1)
q_s<-unique(b$V2)

#d_s = 5000000
#q_s = 1000000

#out_name<-paste(device,".png",sep="")
png("test.png",width=1000)

y_min<-1
x_min<-1000
y_max<-2500
x_max<-100000

plot(1, 1, type="n",log="x",
	ylim=c(y_min,y_max),xlim=c(x_min,x_max),
	pch=1,
	bty="n",xlab="Queue Size",ylab="Run Time (ms)",
	main="Binary Search")

mark<-0
for (d in d_s) {
	b_p<-mat.or.vec(0,3)
	s_p<-mat.or.vec(0,3)
	i_gm_p<-mat.or.vec(0,3)
	i_sm_p<-mat.or.vec(0,3)
	t_gm_p<-mat.or.vec(0,3)
	t_sm_p<-mat.or.vec(0,3)

	s_t_gm_p<-mat.or.vec(0,3)
	s_i_gm_p<-mat.or.vec(0,3)

	for (q in q_s) {
		t_sm_s<-subset(t_sm, V1==d & V2==q)
		t_sm_p<-rbind( t_sm_p, c( d, q, min(t_sm_s$V5)) )

		t_gm_s<-subset(t_gm, V1==d & V2==q)
		t_gm_p<-rbind( t_gm_p, c( d, q, min(t_gm_s$V5)) )

		s_t_gm_s<-subset(s_t_gm, V1==d & V2==q)
		s_t_gm_p<-rbind( s_t_gm_p, c( d, q, min(s_t_gm_s$V5)) )

		i_gm_s<-subset(i_gm, V1==d & V2==q)
		i_gm_p<-rbind( i_gm_p, c( d, q, min(i_gm_s$V5)) )

		i_sm_s<-subset(i_sm, V1==d & V2==q)
		i_sm_p<-rbind( i_sm_p, c( d, q, min(i_sm_s$V5)) )

		s_i_gm_s<-subset(s_i_gm, V1==d & V2==q)
		s_i_gm_p<-rbind( s_i_gm_p, c( d, q, min(s_i_gm_s$V5)) )

		b_s<-subset(b, V1==d & V2==q)
		b_p<-rbind( b_p, c( d, q, min(b_s$V5)) )

		s_s<-subset(s, V1==d & V2==q)
		s_p<-rbind( s_p, c( d, q, min(s_s$V5)) )

#		s_t_gm_s<-subset(s_t_gm, V1==d & V2==q)
#		s_i_gm_s<-subset(s_i_gm, V1==d & V2==q)

#		#bs_s<-subset(bs, V1==d & V2==q)
#		#is_s<-subset(is, V1==d & V2==q)
#		#ts_s<-subset(ts, V1==d & V2==q)

	}
	mark<-mark+1

	lines(b_p[,2], b_p[,3], type="o" , pch=mark, col=1)
	lines(s_p[,2], s_p[,3], type="o" , pch=mark, col=2)
	lines(i_gm_p[,2], i_gm_p[,3], type="o" , pch=mark, col=3)
	lines(i_sm_p[,2], i_sm_p[,3], type="o" , pch=mark, col=4)
	lines(t_gm_p[,2], t_gm_p[,3], type="o" , pch=mark, col=5)
	lines(t_sm_p[,2], t_sm_p[,3], type="o" , pch=mark, col=6)
	lines(s_i_sm_p[,2], s_i_sm_p[,3], type="o" , pch=mark, col=7)
	lines(s_t_gm_p[,2], s_t_gm_p[,3], type="o" , pch=mark, col=8)
}

legend("left", as.character(d_s), pch=1:4, bty="n", lty=1)

b_l<-paste("No Index (",device,")",sep="")
s_l<-paste("Sorted, No Index (",device,")",sep="")
t_sm_l<-paste("Tree, Shared Mem(",device,")",sep="")
t_gm_l<-paste("Tree, Global Mem(",device,")", sep="")
i_sm_l<-paste("List, Shared Mem(",device,")",sep="")
i_gm_l<-paste("List, Global Mem(",device,")",sep="")
s_t_gm_l<-paste("Sorted, Tree, Global Mem(",device,")",sep="")
s_i_gm_l<-paste("Sorted, List, Global Mem(",device,")",sep="")

legend("topleft", c(b_l,
				  s_l,
				  i_gm_l,
				  i_sm_l,
				  t_gm_l,
				  t_sm_l,
				  s_i_gm_l,
				  s_t_gm_l
				 ), bty="n", lty=1, col=1:8)
	
dev.off()


