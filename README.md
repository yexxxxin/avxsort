# avxsort
注明：
目前我们的电脑是基于4核进行测试的，在300w、500w的数据下平均稳定在6-7倍，理论上在更多核的环境里我们的运行时间会得到更大的提升（因为可以同时运行更多的线程，我们的开辟内存、结构体赋值等是并行实现的）
使用方法：
针对avxsort
  编译:
	g++ -g -fopenmp -mavx512f  avxsort.cpp -o avxsort
   运行程序
       	./avxsort<array_size> <num_threads>
	<array_size>：要排序的数组大小。
	<num_threads>：用于排序的线程数。

实例输入：
	./avxsort 3000000 4（这里可以输入各种线程进行测试，我们输入4是一个测试的例子，理论上在具有更多核的环境下，更多的线程数就会带来更好的性能）
实例输出：
	Time taken: 0.442 seconds
	Validation: Array is sorted correctly by score

