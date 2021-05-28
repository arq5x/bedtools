###############
*map*
###############

|

.. image:: ../images/tool-glyphs/map-glyph.png 
    :width: 600pt 
    :align: center

|
``bedtools map`` allows one to map overlapping features in a B file onto 
features in an A file and apply statistics and/or summary operations on those 
features.  

For example, one could use ``bedtools map`` to compute the average
score of BEDGRAPH records that overlap genes. Since the fourth column in 
BEDGRAPH is the score, the following command illustrates how this would be done:

.. code-block:: bash

    $ bedtools map -a genes.bed -b peaks.bedgraph -c 4 -o mean

Another example is discussed in this Biostars 
`post <http://www.biostars.org/p/61653/>`_.


.. note::

    ``bedtools map`` requires each input file to be sorted by genome coordinate.
    For BED files, this can be done with ``sort -k1,1 -k2,2n``.



==========================================================================
Usage and option summary
==========================================================================
**Usage**:
::

  bedtools map [OPTIONS] -a <bed/gff/vcf> -b <bed/gff/vcf>

**(or)**:
::

  mapBed [OPTIONS] -a <bed/gff/vcf> -b <bed/gff/vcf>
  
  
===========================      ===============================================================================================================================================================================================================
Option                           Description
===========================      ===============================================================================================================================================================================================================
**-c**                           | Specify the column from the B file to map onto intervals in A.
                                 | ``Default: 5``

**-o**                           Specify the operation that should be applied to ``-c``.

                                 | Valid operations: 
                                 
                                 | **sum** - *numeric only*
                                 | **count** - *numeric or text*
                                 | **count_distinct** - *numeric or text*
                                 | **min** - *numeric only*
                                 | **max** - *numeric only*
                                 | **mean** - *numeric only*
                                 | **median** - *numeric only*
                                 | **antimode** - *numeric or text*
                                 | **collapse** (i.e., print a comma separated list) - *numeric or text*
                                 | **distinct** (i.e., print a comma separated list) - *numeric or text*
                                 | **concat** (i.e., print a comma separated list) - *numeric or text*
                                 |
                                 | ``Default: sum``

**-f**		                     Minimum overlap required as a fraction of A. Default is 1E-9 (i.e. 1bp).
**-r**		                     Require that the fraction of overlap be reciprocal for A and B. In other words, if -f is 0.90 and -r is used, this requires that B overlap at least 90% of A and that A also overlaps at least 90% of B.
**-s**		                     Force "strandedness". That is, only report hits in B that overlap A on the same strand. By default, overlaps are reported without respect to strand.
**-S**	                         Require different strandedness.  That is, only report hits in B that overlap A on the _opposite_ strand. By default, overlaps are reported without respect to strand.
**-null**                        | The value to print if no overlaps are found for an A interval.
                                 | ``Default: "."``

**-header**	                     Print the header from the A file prior to results.
===========================      ===============================================================================================================================================================================================================



================================================================================
Default behavior - compute the ``sum`` of the ``score`` column for all overlaps.
================================================================================
By default, ``map`` computes the sum of the 5th column (the ``score`` field for
BED format) for all intervals in B that overlap each interval in A.

.. tip::

    Records in A that have no overlap will, by default, return ``.`` for the
    computed value from B.  This can be changed with the ``-null`` option.

.. code-block:: bash

    $ cat a.bed
    chr1	10	20	a1	1	+
    chr1	50	60	a2	2	-
    chr1	80	90	a3	3	-

    $ cat b.bed
    chr1	12	14	b1	2	+
    chr1	13	15	b2	5	-
    chr1	16	18	b3	5	+
    chr1	82	85	b4	2	-
    chr1	85	87	b5	3	+

    $ bedtools map -a a.bed -b b.bed 
    chr1	10	20	a1	1	+	12
    chr1	50	60	a2	2	-	.
    chr1	80	90	a3	3	-	5
    

================================================================================
``mean`` Compute the mean of a column from overlapping intervals
================================================================================

.. code-block:: bash

    $ cat a.bed
    chr1	10	20	a1	1	+
    chr1	50	60	a2	2	-
    chr1	80	90	a3	3	-

    $ cat b.bed
    chr1	12	14	b1	2	+
    chr1	13	15	b2	5	-
    chr1	16	18	b3	5	+
    chr1	82	85	b4	2	-
    chr1	85	87	b5	3	+

    $ bedtools map -a a.bed -b b.bed -c 5 -o mean
    chr1	10	20	a1	1	+	4
    chr1	50	60	a2	2	-	.
    chr1	80	90	a3	3	-	2.5
    
    
================================================================================
``collapse`` List each value of a column from overlapping intervals
================================================================================

.. code-block:: bash

    $ bedtools map -a a.bed -b b.bed -c 5 -o collapse
    chr1	10	20	a1	1	+	2,5,5
    chr1	50	60	a2	2	-	.
    chr1	80	90	a3	3	-	2,3
    

================================================================================
``distinct`` List each *unique* value of a column from overlapping intervals
================================================================================

.. code-block:: bash

    $ bedtools map -a a.bed -b b.bed -c 5 -o distinct
    chr1	10	20	a1	1	+	2,5
    chr1	50	60	a2	2	-	.
    chr1	80	90	a3	3	-	2,3
    
================================================================================
``-s`` Only include intervals that overlap on the *same* strand.
================================================================================

.. code-block:: bash

    $ bedtools map -a a.bed -b b.bed -c 5 -o collapse -s
    chr1	10	20	a1	1	+	2,5
    chr1	50	60	a2	2	-	.
    chr1	80	90	a3	3	-	2
    
================================================================================
``-S`` Only include intervals that overlap on the *opposite* strand.
================================================================================

.. code-block:: bash

    $ bedtools map -a a.bed -b b.bed -c 5 -o collapse -S
    chr1	10	20	a1	1	+	5
    chr1	50	60	a2	2	-	.
    chr1	80	90	a3	3	-	3
