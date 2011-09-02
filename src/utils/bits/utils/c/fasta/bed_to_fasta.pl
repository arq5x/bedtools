#!/usr/bin/perl -w
use strict;
use Getopt::Long;
use File::Basename;
my $prog = basename($0);

sub print_usage {
	my ($msg) = @_;
    warn <<"EOF";

$msg

USAGE
  $prog [options]

DESCRIPTION
  Return a FASTA file with the regions identified by a bed file

  Genome files must be in:
  GENOME/PRE + CHR +POST
  Where genome GENOME is the --genome parameter, pre is the --pre parameter,
  POST is the --post parameter, and CHR is the first field in the bed file

OPTIONS
  -h                      Print this help message
  --bed         file      Path to bed file
  --genome      directory Path to genome files
  --pre         prefix    Genome file prefix
  --post        postfix   Genome file postfix
  --line        length    Genome file line length
  --get_fasta   path      Path to the  get_fasta_interval program

EOF
	exit;
}

my $bed_file_name;
my $genome_path;
my $pre;
my $post;
my $line_len;
my $get_fasta_path;
my $sep = "\t";
my $help = 0;

GetOptions ("bed=s"	=> \$bed_file_name,		# string
			"genome=s"	=> \$genome_path,		# string
			"pre=s"	=> \$pre,		# string
			"post=s"	=> \$post,		# string
			"line=i"	=> \$line_len,
			"get_fasta=s"	=> \$get_fasta_path,		# string
			"h"			=> \$help) or print_usage(); 

print_usage() if $help;

print_usage("No bed file") if not($bed_file_name);
print_usage("No genome path") if not($genome_path);
print_usage("No pre") if not($pre);
print_usage("No post") if not($post);
print_usage("No fasta path") if not($get_fasta_path);
print_usage("No line length") if not($line_len);

open(FILE, $bed_file_name) or die "Could not open $bed_file_name.\n$!";

while (my $l = <FILE>) {
	chomp($l);
	my @a = split ($sep, $l);

	my $chr = $a[0];
	my $start = $a[1];
	my $end = $a[2];

	my $genome_file = $genome_path . "/" . $pre . $chr . $post;
	print ">$l\n";

	if (-e $genome_file) {
		system("$get_fasta_path $start $end $genome_file $line_len");
	}

}
