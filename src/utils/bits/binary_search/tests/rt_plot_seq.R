range<-"5M_1M"
#device<-"IntelXeonE5504"
device<-"IntelCore2X9770"
#device<-"IntelXeonX5550"

b_name<-paste("bsearch_seq-",range,".",device,".txt",sep="")
s_name<-paste("sort_bsearch_seq-",range,".",device,".txt",sep="")
t_name<-paste("t_bsearch_seq-",range,".",device,".txt",sep="")
i_name<-paste("i_bsearch_seq-",range,".",device,".txt",sep="")

b <- read.delim(b_name,header=F,sep=",")
s <- read.delim(s_name,header=F,sep=",")
i <- read.delim(i_name,header=F,sep=",")
t <- read.delim(t_name,header=F,sep=",")

#d_s<-unique(t_sm$V1)
#q_s<-unique(t_sm$V2)

d_s = 5000000
q_s = 1000000

out_name<-paste(device,".png",sep="")
png(out_name,width=1000)
for (d in d_s) {
	for (q in q_s) {

		b_s<-subset(b, V1==d & V2==q)
		s_s<-subset(s, V1==d & V2==q)
		i_s<-subset(i, V1==d & V2==q)
		t_s<-subset(t, V1==d & V2==q)

		x_min<-min(b_s$V3, s_s$V3, i_s$V3, t_s$V3)
		x_max<-max(b_s$V3, s_s$V3, i_s$V3, t_s$V3)
		y_min<-min(b_s$V5, s_s$V5, i_s$V5, t_s$V5)
		y_max<-max(b_s$V5, s_s$V5, i_s$V5, t_s$V5)

		title_string<-paste("Binary Search, Database:",d,
							" Queries:",q,sep="")

		plot(1, 1, type="n",log="x",
			ylim=c(y_min,y_max),xlim=c(x_min,x_max),
			pch=1,
			#col=1,
			bty="n",xlab="Index Size",ylab="Run Time (ms)",
			main=title_string)

		lines(b_s$V3, b_s$V5, type="o" ,pch=1)#,col=5)
		lines(s_s$V3, s_s$V5, type="o" ,pch=2)#,col=6)
		lines(i_s$V3, i_s$V5, type="o" ,pch=3)#,col=6)
		lines(t_s$V3, t_s$V5, type="o" ,pch=4)#,col=6)


		b_l<-paste("No Index (",device,")",sep="")
		s_l<-paste("Sorted, No Index (",device,")",sep="")
		i_l<-paste("List (",device,")",sep="")
		t_l<-paste("Tree (",device,")",sep="")

		legend("left", c( 
						  b_l,
						  s_l,
						  i_l,
						  t_l
						),
				pch=1:4,
				#col=1:8,
				bty="n",
				lty=1)
	}
}
dev.off()


