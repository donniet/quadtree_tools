#ifndef __REDUCE_EQUIVALANCE_HPP__
#define __REDUCE_EQUIVALANCE_HPP__

#include <iterator>

#include <boost/compute/core.hpp>
#include <boost/compute/lambda/placeholders.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>
#include <boost/compute/types/fundamental.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <boost/compute/algorithm/partial_sum.hpp>
#include <boost/compute/algorithm/copy_n.hpp>
#include <boost/compute/algorithm/any_of.hpp>

namespace compute = boost::compute;
using compute::ulong_;
using compute::lambda::_1;


// assumes device iterators
template<typename IteratorFrom, typename IteratorTo>
inline IteratorFrom reduce_equivalances(
  IteratorFrom from_first, IteratorFrom from_last,
  IteratorTo to_first,
  compute::command_queue & queue)
{
  typedef typename std::iterator_traits<IteratorFrom>::difference_type difference_type;
  typedef typename std::iterator_traits<IteratorFrom>::value_type value_type;

  static const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    ulong binary_search(
      __global ulong * arr,
      ulong first,
      ulong last,
      const ulong x
    ) {
      ulong original_last = last;
      while(first < last) {
        ulong m = (first + last) >> 1;
        if (arr[m] > x) {
          last = m;
        } else if (arr[m] < x) {
          first = m + 1;
        } else {
          return m;
        }
      }
      return original_last;
    }

    __kernel void forward_decay(
      __global ulong * from,
      __global ulong * to,
      const ulong count,
      __global ulong * updated
    ) {
      ulong lid = get_local_id(0);
      ulong local_size = get_local_size(0);

      for(ulong l = 1 + lid; l < count; l += local_size) {
        if (from[l-1] == from[l] && to[l-1] != to[l]) {
          updated[l] = to[l-1];
        } else {
          updated[l] = from[l];
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for(ulong l = 1 + lid; l < count; l += local_size) {
        if (updated[l] != from[l]) {
          from[l] = updated[l];
          updated[l] = 1;
        } else {
          updated[l] = 0;
        }
      }
    }

    __kernel void adjacent_equal_key_value(
      __global ulong * keys,
      __global ulong * values,
      const ulong count,
      __global ulong * equal
    ) {
      ulong gid = get_local_id(0);

      if (gid == 0 || gid >= count) return;

      if (keys[gid] == keys[gid-1] && values[gid] == values[gid-1]) {
        equal[gid] = 1;
      } else {
        equal[gid] = 0;
      }
    }

    __kernel void unique_key_value_with_adjacent_equal(
      __global ulong * keys,
      __global ulong * values,
      const ulong count,
      __global ulong * equal_sum,
      __global ulong * temp_keys,
      __global ulong * temp_values
    ) {
      ulong gid = get_local_id(0);
      ulong diff = equal_sum[gid];

      if (gid == 0) return;


      if (diff != equal_sum[gid-1]) {
        // this is a duplicate, move it to the end
        temp_keys[count - diff] = keys[gid];
        temp_values[count - diff] = values[gid];
      } else {
        // else move it back by diff
        temp_keys[gid - diff] = keys[gid];
        temp_values[gid - diff] = values[gid];
      }

      barrier(CLK_GLOBAL_MEM_FENCE);

      gid = get_local_id(0);
      diff = equal_sum[gid];

      keys[gid] = temp_keys[gid];
      values[gid] = temp_values[gid];
    }

    __kernel void update_unique_rules(
      __global ulong * keys,
      __global ulong * values,
      const ulong count,
      __global ulong * updated
    ) {
      ulong lid = get_local_id(0);
      ulong local_size = get_local_size(0);

      for(ulong l = lid; l < count; l += local_size) {
        ulong dex = binary_search(keys, 0, count, values[l]);
        if (dex != count) {
          updated[l] = values[dex];
        } else {
          updated[l] = values[l];
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for(ulong l = lid; l < count; l += local_size) {
        if (updated[l] != values[l]) {
          values[l] = updated[l];
          updated[l] = 1;
        } else {
          updated[l] = 0;
        }
      }
    }
  );

  compute::context context = queue.get_context();
  compute::device device = queue.get_device();
  auto compute_units = device.compute_units();

  difference_type n = std::distance(from_first, from_last);
  IteratorTo to_last = to_first;
  std::advance(to_last, n);

  compute::vector<ulong_> dev_output(n, context);
  compute::vector<ulong_> dev_temp0(n, context);
  compute::vector<ulong_> dev_temp1(n, context);

  compute::program prog = compute::program::build_with_source(source, context);

  compute::kernel forward_decay(prog, "forward_decay");
  compute::kernel adjacent_equal_key_value(prog, "adjacent_equal_key_value");
  compute::kernel unique_key_value_with_adjacent_equal(prog, "unique_key_value_with_adjacent_equal");
  compute::kernel update_unique_rules(prog, "update_unique_rules");
  forward_decay.set_args(from_first.get_buffer(), to_first.get_buffer(), ulong_(n), dev_output);
  adjacent_equal_key_value.set_args(from_first.get_buffer(), to_first.get_buffer(), ulong_(n), dev_output);
  unique_key_value_with_adjacent_equal.set_args(from_first.get_buffer(), to_first.get_buffer(), ulong_(n), dev_output, dev_temp0, dev_temp1);
  update_unique_rules.set_args(from_first.get_buffer(), to_first.get_buffer(), ulong_(n), dev_output);

  bool updated = true;
  while(updated) {
    compute::sort_by_key(from_first, from_first + n, to_first, queue);

    adjacent_equal_key_value.set_arg(2, ulong_(n));
    queue.enqueue_1d_range_kernel(adjacent_equal_key_value, 0, n, 0);

    compute::partial_sum(dev_output.begin(), dev_output.begin() + n, dev_output.begin(), queue);

    unique_key_value_with_adjacent_equal.set_arg(2, ulong_(n));
    queue.enqueue_1d_range_kernel(unique_key_value_with_adjacent_equal, 0, n, 0);

    ulong_ removed;
    compute::copy_n(dev_output.begin() + n - 1, 1, &removed, queue);

    n -= removed;
    from_last = from_first + n;

    forward_decay.set_arg(2, ulong_(n));
    queue.enqueue_1d_range_kernel(forward_decay, 0, compute_units, compute_units);

    updated = compute::any_of(dev_output.begin(), dev_output.begin() + n, _1 == 1, queue);
  }

  updated = true;
  while(updated) {
    update_unique_rules.set_arg(2, ulong_(n));
    queue.enqueue_1d_range_kernel(update_unique_rules, 0, compute_units, compute_units);

    updated = compute::any_of(dev_output.begin(), dev_output.begin() + n, _1 == 1, queue);
  }

  return from_last;
}


#endif // __REDUCE_EQUIVALANCE_HPP__
