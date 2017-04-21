#ifndef __QUADTREE_MERGE_HPP__
#define __QUADTREE_MERGE_HPP__

/*
quadtree should be a list of 64-bit integers in this format:

depth:6
interleaved:58

where interleaved is the bits of y and x interleaved.  There is only one valid
value where depth==0 which is interleaved==0 also.  At depth==1, interleaved
can have the values 00, 01, 10, 11.  At depth two interleaved would use 4 bits,
etc.
*/

template<typename InputIterator, typename OutputIterator>
inline OutputIterator merge_quadtree(
  InputIterator tree_first, InputIterator tree_last,
  OutputIterator merged_first
) {
  static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(

  );
}

#endif
