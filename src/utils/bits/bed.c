#include "bed.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

//{{{ struct interval_node *new_interval_node(unsigned int start,
struct interval_node *new_interval_node(unsigned int start,
										unsigned int end) 
{

   struct interval_node *i = (struct interval_node *)
		malloc( sizeof(struct interval_node) );
	i->start = start;
	i->end = end;
	i->next = NULL;

	return i;
}	
//}}}

//{{{struct chr_list new_chr_list(char *name) {
struct chr_list new_chr_list(char *name)
{
	struct chr_list c;
	c.name = (char *) malloc( sizeof(char) * strlen(name) );
	strcpy(c.name, name);
	c.head = NULL;
	c.size = 0;

	return c;
}
//}}}

//{{{void chr_list_insert_interval(struct chr_list *list,
void chr_list_insert_interval(struct chr_list *list,
							  unsigned int start,
							  unsigned int end)
{
   struct interval_node *i = new_interval_node(start, end);
   chr_list_insert_interval_node(list, i);
}
//}}}

//{{{void chr_list_insert_interval_node(struct chr_list *list, 
void chr_list_insert_interval_node(struct chr_list *list, 
								   struct interval_node *new_interval_node)
{
	new_interval_node->next = list->head;
	list->head = new_interval_node;
	list->size++;
}
//}}}

//{{{int compare_chr_lists (const void *a, const void *b)
int compare_chr_lists (const void *a, const void *b)
{
	struct chr_list *a_i = (struct chr_list *)a;
	struct chr_list *b_i = (struct chr_list *)b;
	return strcmp(a_i->name, b_i->name);
}
//}}}

//{{{ int compare_interval_node_by_start(const void *a, const void *b) {
int compare_interval_node_by_start(const void *a, const void *b) {
	struct interval_node *a_i = (struct interval_node *)a;
	struct interval_node *b_i = (struct interval_node *)b;
	return a_i->start - b_i->start;
}
//}}}

//{{{ int compare_interval_by_start(const void *a, const void *b)
int compare_interval_by_start(const void *a, const void *b)
{
	struct interval *a_i = (struct interval *)a;
	struct interval *b_i = (struct interval *)b;
	return a_i->start - b_i->start;
}
//}}}

//{{{ int compare_interval_node_by_end(const void *a, const void *b)
int compare_interval_node_by_end(const void *a, const void *b)
{
	struct interval_node *a_i = (struct interval_node *)a;
	struct interval_node *b_i = (struct interval_node *)b;
	//printf("%d\t%d\n", a_i->end, b_i->end);
	return a_i->end - b_i->end;
}
//}}}

//{{{void parse_bed_file(FILE *bed_file, struct chr_list chroms[],
/**
 * parse_bed_file takes three parameters:
 * bed_file is a pointer to an open bed file, which is a a tab-delmited file
 * where column one is the chromosome nubmer (e.g., "chr1", ..., "chrY"),
 * column two is the start offset of the interval, and column three is the end
 * end of the interval
 * chr_list is an array of chr_lists where there is an entry in chr_list
 * correspoinding to any of the chromonosomes that occur in bed_file.  If there
 * is an entry in bed_file that does not have a corresponding entry in the
 * chr_list then it will be silently ignored
 * chrom_num is the number of chormosomes represented in chr_list
 *
 * Each lines that start with "chr" 
 */
void parse_bed_file(FILE *bed_file, struct chr_list chroms[], int chrom_num)
{
	char line[LINE_MAX];
	while(fgets(line, LINE_MAX, bed_file) != NULL) {
		if ( strncmp("chr", line, 3) == 0 ) {
			char *chr = strtok(line, "\t");
			char *start = strtok(NULL, "\t");
			char *end = strtok(NULL, "\t");
			
			struct chr_list find_node = new_chr_list(chr);
			struct chr_list *curr = (struct chr_list *) bsearch(
					(void *) (&find_node), chroms, chrom_num, 
					sizeof(struct chr_list), compare_chr_lists);
			
			if (curr != NULL)
				chr_list_insert_interval(curr, atoi(start), atoi(end));
			//else
				//printf("Not found: %s\n", chr);
		}
	}
}
//}}}

//{{{ int parse_bed_line(FILE *bed_file,
int parse_bed_line(FILE *bed_file,
				   char *chr, 
				   unsigned int *start,
				   unsigned int *end)
{
	char line[LINE_MAX];
	while(fgets(line, LINE_MAX, bed_file) != NULL) {
		if ( strncmp("chr", line, 3) == 0 ) {
			strcpy(chr, strtok(line, "\t"));
			*start = atoi( strtok(NULL, "\t") );
			*end = atoi( strtok(NULL, "\t") );

			return 1;
		}
	}

	return 0;
}
//}}}

//{{{void free_chr_list(struct chr_list *list, int chrom_num) 
void free_chr_list(struct chr_list *list, int chrom_num) 
{
	int i;
	for (i = 0; i < chrom_num; i++) {
		struct interval_node *head = list[i].head, *next = NULL;

		if (head != NULL) {
			while (head->next != NULL) {
				next = head->next;
				head->next = next->next;
				free(next);
			}
			free(head);
		}
	}
	//free(list);
}
//}}}

//{{{int chr_name_to_int(char *name)
int chr_name_to_int(char *name)
{
	int chr = 0;

	if (name[3] == 'X')
		chr = 23;
	else if (name[3] == 'Y')
		chr = 24;
	else 
		chr = atoi(name + 3);	

	return chr;
}
//}}}

//{{{int parse_bed_file_by_line(char *bed_file_name, struct bed_line **lines) {
int parse_bed_file_by_line(char *bed_file_name, struct bed_line **lines)
{
	
	FILE *bed_file = fopen(bed_file_name, "r");
	char line[LINE_MAX];
	int line_num = 0;
	while(fgets(line, LINE_MAX, bed_file) != NULL) {
		if ( strncmp("chr", line, 3) == 0 )
			line_num++;
	}
	fclose(bed_file);

	*lines = (struct bed_line *) malloc (line_num * sizeof(struct bed_line));

	bed_file = fopen(bed_file_name, "r");
	int i = 0;
	while(fgets(line, LINE_MAX, bed_file) != NULL) {
		if ( strncmp("chr", line, 3) == 0 ) {
			char *chr_s = strtok(line, "\t");
			int chr = 0;
			if (chr_s[3] == 'X')
				chr = 23;
			else if (chr_s[3] == 'Y')
				chr = 24;
			else {
				//char t[3];
				//strcpy(t, chr_s + 3);
				chr = atoi(chr_s + 3);	
			}
			int start = atoi(strtok(NULL, "\t"));
			int end = atoi(strtok(NULL, "\t"));
			(*lines)[i].chr = chr;
			(*lines)[i].start = start;
			(*lines)[i].end = end;
			++i;
		}
	}
	fclose(bed_file);

	return line_num;
}
//}}}

//{{{int chr_list_from_bed_file(struct chr_list **list, char **chrom_names,
int chr_list_from_bed_file(struct chr_list **list, char **chrom_names,
		int chrom_num, char *bed_file_name) 
{
	*list = (struct chr_list *) malloc(sizeof(struct chr_list) * chrom_num);

	// initialize chrom lists
	int i;
	for (i = 0; i < chrom_num; i++) {
		(*list)[i] = new_chr_list(chrom_names[i]);
	}

	// chr_lists need to be sorted before used
	qsort(*list, chrom_num, sizeof(struct chr_list), compare_chr_lists);

	FILE *file = fopen(bed_file_name, "r");

	if (file == NULL) {
		fprintf(stderr, "Could not open file:%s\n", bed_file_name);
		return 1;
	}

	parse_bed_file(file, *list, chrom_num);

	fclose(file);

	return 0;
}
//}}}

//{{{int trim(struct chr_list *universe, struct chr_list *interval_set, 
int trim(struct chr_list *universe, struct chr_list *interval_set, 
		 int chrom_num) 
{
	int i, c = 0;
	for (i = 0; i < chrom_num; i++) {
		struct interval_node *curr_chrm;

		struct interval_node *curr = interval_set[i].head, *last = NULL;

		while (curr != NULL) {
			curr_chrm = universe[i].head;
			//int offset = -1, start = -1;
			while (curr_chrm != NULL) {
				if (	(curr->start >= curr_chrm->start) &&
						(curr->end <= curr_chrm->end) ) {

					break;

				// Trim end
				} else if (	(curr->start >= curr_chrm->start) &&
							(curr->start <= curr_chrm->end) &&
							(curr->end > curr_chrm->end) ) {
					curr->end = curr_chrm->end;
					break;
				// Trim start
				} else if (	(curr->end >= curr_chrm->start) &&
							(curr->end <= curr_chrm->end) &&
							(curr->start < curr_chrm->start) ) {
					curr->start = curr_chrm->start;
					break;
				// Trim both
				} else if (	(curr->start <= curr_chrm->start) &&
							(curr->end >= curr_chrm->end) ) {
					curr->start = curr_chrm->start;
					curr->end = curr_chrm->end;
				}

				curr_chrm = curr_chrm->next;
			}
		/* 
		 * if the current interval is outside the universe, remove it and drop
		 * the size by one
		 */
			if (curr_chrm == NULL) {
				//printf(stderr, "rem\n");
				interval_set[i].size = interval_set[i].size - 1;
				c++;
				if (curr == interval_set[i].head) {
					interval_set[i].head = curr->next;
					free(curr);
					curr = interval_set[i].head;
				} else {
					last->next = curr->next;
					free(curr);
					curr = last->next;
				}
			} else {
				last = curr;
				curr = curr->next;
			}
		}
	}

	return c;
}
//}}}

//{{{int chr_array_from_list(struct chr_list *list, struct bed_line **array, 
int chr_array_from_list(struct chr_list *list, struct bed_line **array, 
		int chrom_num)
{
	// find out how many intervals there are
	int i, total_size = 0;
	for (i = 0; i < chrom_num; i++) {
		total_size += list[i].size;
	}

	*array = (struct bed_line *) malloc( sizeof(struct bed_line) * total_size);

	int j = 0;
	for (i = 0; i < chrom_num; i++) {
		//struct interval_node *curr = list[i].head, *last = NULL;
		struct interval_node *curr = list[i].head;

		int chr = chr_name_to_int(list[i].name);

		while (curr != NULL) {
			(*array)[j].chr = chr;
			(*array)[j].start = curr->start;
			(*array)[j].end = curr->end;
			(*array)[j].offset = curr->offset;
			j++;
			curr = curr->next;
		}
	}

	return total_size;
}
//}}}
