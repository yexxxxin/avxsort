
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include<omp.h>
#include <cstdlib>
#include <stack>
#include <immintrin.h>
#include <ctime>
#include<random>
#include <tuple>
#include<cstring>
#include <unordered_map>
struct Data {
    int id;
    int age;
    float weight;
    float height; // Additional attribute
    std::string name;
    std::string address;
    std::string email;
    std::string phone;
    std::string city;
    std::string country;
    int salary;
    int years_of_experience; // Additional attribute
    bool employed; // Additional attribute
    float rating; // Additional attribute
    float score; // Additional attribute
};


void insertionSort(float* data, int* indices, int start, int end) {
    int count = end - start + 1;
    if (count <= 1)
    {
        return;
    }
    long i, j;
    for (i = start; i < start + count; ++i) {
        float key = data[i];
        int keyIndex = indices[i];

        for (j = i; j >= 1 && key < data[j - 1]; j--) {
            data[j] = data[j - 1];
            indices[j] = indices[j - 1];
        }
        data[j] = key;
        indices[j] = keyIndex;
    }
}


void partition_16(float*& in, int*& index,
    __m512& PIVOT1,
    __m512& PIVOT2,
    float*& bottom,
    float*& middle,
    float*& top,
    int*& bottom_index,
    int*& middle_index,
    int*& top_index,
    int offset)
{


    __m512 INPUT;
    __m512i INDEX;
    __mmask16 MED1;
    __mmask16 HIGH1;
    __m512 LOW;
    __m512 HIGH;
    __m512 MED;
    __m512i LOWi;
    __m512i HIGHi;
    __m512i MEDi;


    INPUT = _mm512_loadu_ps(in + offset);
    INDEX = _mm512_loadu_epi32(index + offset); 
    
    MED1 = _mm512_cmp_ps_mask(INPUT, PIVOT1, _CMP_LE_OQ);  
    HIGH1 = _mm512_cmp_ps_mask(INPUT, PIVOT2, _CMP_GE_OQ); 
    __mmask16 mmask = 0xFFFF & (~MED1 & ~HIGH1); 


    LOW = _mm512_mask_compress_ps(INPUT, MED1, INPUT);
    MED = _mm512_mask_compress_ps(INPUT, mmask, INPUT);
    HIGH = _mm512_mask_compress_ps(INPUT, HIGH1, INPUT);
    LOWi = _mm512_mask_compress_epi32(INDEX, MED1, INDEX);
    MEDi = _mm512_mask_compress_epi32(INDEX, mmask, INDEX);
    HIGHi = _mm512_mask_compress_epi32(INDEX, HIGH1, INDEX);

    _mm512_storeu_ps(bottom, LOW);
    _mm512_storeu_ps(middle, MED);
    _mm512_storeu_ps(top, HIGH);
    _mm512_storeu_epi32(bottom_index, LOWi);
    _mm512_storeu_epi32(middle_index, MEDi);
    _mm512_storeu_epi32(top_index, HIGHi);

    /*    float data1[16];
        _mm512_store_ps(data1, PIVOT);
        std::cout << "PIVOT values:" << std::endl;
        for (int i = 0; i < 16; ++i) {
            std::cout << data1[i] << " ";
        }
        std::cout << std::endl;

        int b = __builtin_popcount(MED1);
        int m = __builtin_popcount(mmask);
        int t = __builtin_popcount(HIGH1);
        float data2[16];
        _mm512_store_ps(data2, LOW);
        std::cout << "LOW values:" << std::endl;
        for (int i = 0; i < b; ++i) {
            std::cout << data2[i] << " ";
        }
        std::cout << std::endl;
        float data3[16];
        _mm512_store_ps(data3, MED);
        std::cout << "MED values:" << std::endl;
        for (int i = 0; i < m; ++i) {
            std::cout << data3[i] << " ";
        }
        std::cout << std::endl;
        float data4[16];
        _mm512_store_ps(data4, HIGH);
        std::cout << "TOP values:" << std::endl;
        for (int i = 0; i < t; ++i) {
            std::cout << data4[i] << " ";
        }
        std::cout << std::endl;*/
       
    int b = __builtin_popcount(MED1);
    int m = __builtin_popcount(mmask);
    int t = __builtin_popcount(HIGH1);
    bottom += b;
    middle += m;
    top += t;
    bottom_index += b;
    middle_index += m;
    top_index += t;

}

void partition_16(float*& in, int*& index,
    __m512& PIVOT,
    float*& bottom,
    float*& middle,
    float*& top,
    int*& bottom_index,
    int*& middle_index,
    int*& top_index,
    int offset)
{


    __m512 INPUT;
    __m512i INDEX;
    __mmask16 MED1;
    __mmask16 HIGH1;
    __m512 LOW;
    __m512 HIGH;
    __m512 MED;
    __m512i LOWi;
    __m512i HIGHi;
    __m512i MEDi;

    

    INPUT = _mm512_loadu_ps(in + offset);
    INDEX = _mm512_loadu_epi32(index + offset);  
   
    MED1 = _mm512_cmp_ps_mask(INPUT, PIVOT, _CMP_LT_OQ);  
    HIGH1 = _mm512_cmp_ps_mask(INPUT, PIVOT, _CMP_GT_OQ); 
    __mmask16 mmask = 0xFFFF & (~MED1 & ~HIGH1); 


    LOW = _mm512_mask_compress_ps(INPUT, MED1, INPUT);
    MED = _mm512_mask_compress_ps(INPUT, mmask, INPUT);
    HIGH = _mm512_mask_compress_ps(INPUT, HIGH1, INPUT);
    LOWi = _mm512_mask_compress_epi32(INDEX, MED1, INDEX);
    MEDi = _mm512_mask_compress_epi32(INDEX, mmask, INDEX);
    HIGHi = _mm512_mask_compress_epi32(INDEX, HIGH1, INDEX);

    _mm512_storeu_ps(bottom, LOW);
    _mm512_storeu_ps(middle, MED);
    _mm512_storeu_ps(top, HIGH);
    _mm512_storeu_epi32(bottom_index, LOWi);
    _mm512_storeu_epi32(middle_index, MEDi);
    _mm512_storeu_epi32(top_index, HIGHi);

    /*    float data1[16];
        _mm512_store_ps(data1, PIVOT);
        std::cout << "PIVOT values:" << std::endl;
        for (int i = 0; i < 16; ++i) {
            std::cout << data1[i] << " ";
        }
        std::cout << std::endl;

        int b = __builtin_popcount(MED1);
        int m = __builtin_popcount(mmask);
        int t = __builtin_popcount(HIGH1);
        float data2[16];
        _mm512_store_ps(data2, LOW);
        std::cout << "LOW values:" << std::endl;
        for (int i = 0; i < b; ++i) {
            std::cout << data2[i] << " ";
        }
        std::cout << std::endl;
        float data3[16];
        _mm512_store_ps(data3, MED);
        std::cout << "MED values:" << std::endl;
        for (int i = 0; i < m; ++i) {
            std::cout << data3[i] << " ";
        }
        std::cout << std::endl;
        float data4[16];
        _mm512_store_ps(data4, HIGH);
        std::cout << "TOP values:" << std::endl;
        for (int i = 0; i < t; ++i) {
            std::cout << data4[i] << " ";
        }
        std::cout << std::endl;*/
       
    int b = __builtin_popcount(MED1);
    int m = __builtin_popcount(mmask);
    int t = __builtin_popcount(HIGH1);
    bottom += b;
    middle += m;
    top += t;
    bottom_index += b;
    middle_index += m;
    top_index += t;

}

void partition_avx512(float*& in, int*& index,
    int start,
    int end,
    unsigned long& n,
    float pivot1,
    float pivot2,
    float*& bottom,
    float*& middle,
    float*& top,
    int*& bottom_index,
    int*& middle_index,
    int*& top_inddex,
    int& offset)
{
    __m128 a = _mm_set_ps1(pivot1);
    __m128 b = _mm_set_ps1(pivot2);
    __m512 PIVOT1 = _mm512_broadcastss_ps(a);
    __m512 PIVOT2 = _mm512_broadcastss_ps(b);


    while(n>=16){
        partition_16(in, index, PIVOT1, PIVOT2, bottom, middle, top, bottom_index, middle_index, top_inddex, offset);
        offset+=16;
        n-=16;

    }

}

void partition_avx512(float*& in, int*& index,
    int start,
    int end,
    unsigned long& n,
    float pivot,
    float*& bottom,
    float*& middle,
    float*& top,
    int*& bottom_index,
    int*& middle_index,
    int*& top_inddex,
    int& offset)
{
    __m128 a = _mm_set_ps1(pivot);
    __m512 PIVOT = _mm512_broadcastss_ps(a);




    while(n>=16) {
        partition_16(in, index, PIVOT, bottom, middle, top, bottom_index, middle_index, top_inddex, offset);
        offset+=16;
        n-=16;
        
    }

}



void Partition(float* data, int* index, int start, int end,
    float* tmp2, float* tmp3, int* index2, int* index3,
    unsigned long& out_bottom_offset,
    unsigned long& out_bottom_count,
    unsigned long& out_middle_offset,
    unsigned long& out_middle_count,
    unsigned long& out_top_offset,
    unsigned long& out_top_count)  
{
    int count = end - start + 1;
    float* bottom = data + start;
    float* middle = tmp2;
    float* top = tmp3;
    int* bottom_index = index + start;
    int* middle_index = index2;
    int* top_index = index3;
    //此时n的值为数组长度
    unsigned long n = count;
    


    int
        quarter = n / 4,
        i1 = 0,
        i5 = n - 1,
        i3 = n / 2,
        i2 = i3 - quarter,
        i4 = i3 + quarter;
    float
        e1 = data[i1+start],
        e2 = data[i2+start],
        e3 = data[i3+start],
        e4 = data[i4+start],
        e5 = data[i5+start];

    float t; 

    
    if (e1 > e2)
        t = e1, e1 = e2, e2 = t;
    if (e4 > e5)
        t = e4, e4 = e5, e5 = t;
    if (e1 > e3)
        t = e1, e1 = e3, e3 = t;
    if (e2 > e3)
        t = e2, e2 = e3, e3 = t;
    if (e1 > e4)
        t = e1, e1 = e4, e4 = t;
    if (e3 > e4)
        t = e3, e3 = e4, e4 = t;
    if (e2 > e5)
        t = e2, e2 = e5, e5 = t;
    if (e2 > e3)
        t = e2, e2 = e3, e3 = t;
    if (e4 > e5)
        t = e4, e4 = e5, e5 = t;

   
    float pivot1 = e2, pivot2 = e4;

    //std::cout <<"pivot1的值是" << pivot1 << std::endl;
    //此时偏移量为start
    int offset = start;
    bool const single_pivoted = (pivot1 == pivot2);
    if (!single_pivoted) {
        partition_avx512(data, index, start, end, n, pivot1, pivot2, bottom, middle, top, bottom_index, middle_index, top_index, offset);
    }
    else {
        partition_avx512(data, index, start, end, n, pivot1, bottom, middle, top, bottom_index, middle_index, top_index, offset);
    }
    if (!single_pivoted) {
        //处理不满16的情况
        while (n > 0)
        {
            if (data[offset] <= pivot1)
            {
                bottom[0] = data[offset];
                bottom += 1;
                n--;
                bottom_index[0] = index[offset];
                bottom_index++;
                offset++;
                //std::cout << "bottom" << data[offset - 1] << " ";
            }
            else if (data[offset] >= pivot2)
            {
                top[0] = data[offset];
                top += 1;
                n--;
                top_index[0] = index[offset];
                top_index++;
                offset++;
                //std::cout << "top" << data[offset - 1] << " ";
            }
            else {
                middle[0] = data[offset];
                middle += 1;
                n--;
                middle_index[0] = index[offset];
                middle_index++;
                offset++;
                //std::cout << "middle" << data[offset - 1] << " ";
            }
        }
    }
    else {
        //处理不满16的情况
        while (n > 0)
        {
            if (data[offset] < pivot1)
            {
                bottom[0] = data[offset];
                bottom += 1;
                n--;
                bottom_index[0] = index[offset];
                bottom_index++;
                offset++;
                //std::cout << "bottom" << data[offset - 1] << " ";
            }
            else if (data[offset] > pivot1)
            {
                top[0] = data[offset];
                top += 1;
                n--;
                top_index[0] = index[offset];
                top_index++;
                offset++;
                //std::cout << "top" << data[offset - 1] << " ";
            }
            else {
                middle[0] = data[offset];
                middle += 1;
                n--;
                middle_index[0] = index[offset];
                middle_index++;
                offset++;
                //std::cout << "middle" << data[offset - 1] << " ";
            }

        }
    }

    //将bottom和top里的数全部转到原数组里面
    int bottom_count = ((uint64_t)bottom - (uint64_t)(data + start)) / sizeof(float);
    int middle_count = ((uint64_t)middle - (uint64_t)tmp2) / sizeof(float);
    int top_count = ((uint64_t)top - (uint64_t)tmp3) / sizeof(float);


    if (middle_count > 0)
    {
        memcpy(data + start + bottom_count, tmp2, middle_count * sizeof(float));
        memcpy(index + start + bottom_count, index2, middle_count * sizeof(int));
    }
    if (top_count > 0)
    {
        memcpy(data + start + bottom_count + middle_count, tmp3, top_count * sizeof(float));
        memcpy(index + start + bottom_count + middle_count, index3, top_count * sizeof(int));
    }
    out_bottom_offset = 0;
    out_bottom_count = bottom_count;
    out_middle_offset = bottom_count;
    out_middle_count = single_pivoted ? 0 : middle_count;
    out_top_offset = bottom_count + middle_count;
    out_top_count = top_count;
    while (out_bottom_count > 0 && data[start + out_bottom_count - 1] == pivot1)
    {
        --out_bottom_count;
    }
    while (out_top_count > 0 && data[start + out_top_offset] == pivot2)
    {
        ++out_top_offset;
        --out_top_count;
    }
    while (out_middle_count > 0 && data[start + out_middle_offset] == pivot1)
    {
        ++out_middle_offset;
        --out_middle_count;
    }
    while (out_middle_count > 0 && data[start + out_middle_count - 1] == pivot2)
    {
        --out_middle_count;
    }


}


void double_quickSort(float* data, int* index, int start, int end, int depth)
{
    if (end - start + 1 > 32)
    {
        int count = end - start + 1;
        unsigned long
            low_offset,
            low_count,
            middle_offset,
            middle_count,
            high_offset,
            high_count;
        //每一个partition中开辟一个内存空间，partition执行完释放
        float* const tmp2 = (float*)malloc(sizeof(float) * count);
        float* const tmp3 = (float*)malloc(sizeof(float) * count);
        int* const index2 = (int*)malloc(sizeof(int) * count);
        int* const index3 = (int*)malloc(sizeof(int) * count);
        //划分区间
        Partition(data, index, start, end, tmp2, tmp3, index2, index3, 
            low_offset,
            low_count,
            middle_offset,
            middle_count,
            high_offset,
            high_count);
        free(tmp2);
        free(tmp3);
        free(index2);
        free(index3);
        {
            if (depth < 6)
            {
#pragma omp task shared(data)
                double_quickSort(data, index, start, low_count + start-1, depth + 1);
#pragma omp task shared(data)
                double_quickSort(data, index, middle_offset + start, middle_count+ middle_offset + start-1, depth + 1);
#pragma omp task shared(data)
                double_quickSort(data, index, high_offset + start, end, depth + 1);
            }
            else {
                double_quickSort(data, index, start, low_count + start-1, depth);
                double_quickSort(data, index, middle_offset + start, middle_count+ middle_offset + start-1, depth);
                double_quickSort(data, index, high_offset + start, end, depth);
            }
        }
    }
    else {
        insertionSort(data, index, start, end);
    }
}


template<typename T>
bool validate_score(const std::vector<T>& array) {
    for (size_t i = 1; i < array.size(); i++) {
        if (array[i].score < array[i - 1].score)
            return false; // Array is not sorted correctly
    }
    return true; // Array is sorted correctly
}
template<typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}
float generateRandomFloat(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}


void avx_sort(float* data, int* index,int array_size,int thread_count)
{
#pragma omp parallel num_threads(thread_count)
    {
#pragma omp single nowait
        {
            double_quickSort(data, index, 0, array_size-1, 0);
        }
    }
}

bool compare_score(const Data& a, const Data& b) {
    return a.score < b.score;
}
void reorder_array(Data* array, Data* sortedArray, const int* index, int array_size) {
    #pragma omp parallel for
    for (int i = 0; i < array_size; ++i) {
        memcpy(&array[i], &sortedArray[index[i]], sizeof(Data));
    }
}
/*void deep_copy(struct Data *dest, const struct Data *src) {

    memcpy(dest, src, sizeof(struct Data));

    // Manually deep copy pointer members
    if (src->name != NULL) {
        dest->name = strdup(src->name);
    } else {
        dest->name = NULL;
    }
    // Copy other members similarly if they are pointers
}
*/
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <array_size> <thread_count>" << std::endl;
        return 1;
    }
    int thread_count = strtol(argv[2], NULL, 10);
    uint64_t array_size = std::strtoull(argv[1], nullptr, 10);
    if (array_size <= 0) {
        std::cout << "Array size must be a positive integer" << std::endl;
        return 1;
    }
    
    std::vector<Data> array(array_size);
    srand(time(NULL)); // Seed the random number generator
    for (uint64_t i = 0; i < array_size; i++) {
        array[i].id = i + 1;
        array[i].age = rand() % 100 + 1; // Random age between 1 and 100
        array[i].weight = static_cast<float>(rand()) / RAND_MAX * 100.0f; // Random weight between 0 and 100
        array[i].height = static_cast<float>(rand()) / RAND_MAX * 200.0f; // Random height between 0 and 200
        array[i].name = "Name" + std::to_string(i);
        array[i].address = "Address" + std::to_string(i);
        array[i].email = "Email" + std::to_string(i) + "@example.com";
        array[i].phone = "+1234567890" + std::to_string(i);
        array[i].city = "City" + std::to_string(i);
        array[i].country = "Country" + std::to_string(i);
        array[i].years_of_experience = rand() % 30; // Random years of experience between 0 and 29
        array[i].employed = rand() % 2; // Random employed status (0 or 1)
        array[i].rating = static_cast<float>(rand()) / RAND_MAX * 5.0f; // Random rating between 0 and 5
        array[i].score = static_cast<float>(rand()) / RAND_MAX * 11000000.0f - 1000000.0f; // Random score between -1000000 and 10000000

    }




  
    auto start = std::chrono::high_resolution_clock::now();
      Data*a=(Data*)malloc(sizeof(Data)*array_size);
    float*b=(float*)malloc(sizeof(float)*array_size);
    int *c=(int *)malloc(sizeof(int)*array_size);
    Data* sortedArray = static_cast<Data*>(a);
    float *scores=static_cast<float*>(b);
    int *index=static_cast<int*>(c);
    #pragma omp parallel for num_threads(thread_count) schedule(dynamic, array_size/thread_count)
    for (int i = 0; i < array_size; i++) {
       new(&scores[i])float(std::move(array[i].score)); 
       new(&index[i])int(std::move(i)); 
       new(&sortedArray[i]) Data(std::move(array[i]));// 初始化 sortedArray
 
    }

    avx_sort(scores, index,array_size,thread_count);
    
#pragma omp parallel for num_threads(thread_count) schedule(dynamic, array_size/thread_count)
 for (int i = 0; i < array_size; i++) {
      array[i]=std::move(sortedArray[index[i]]); // 重新排序 array
  }



    auto end = std::chrono::high_resolution_clock::now();
    



    std::chrono::duration<double> elapsed = end - start;
    std::cout << "sort_time: " << elapsed.count() << " s\n";


   
    
   if (validate_score(array))
        std::cout << "Validation: Array is sorted correctly by score" << std::endl;
   else
        std::cout << "Validation: Array is not sorted correctly by score" << std::endl;
        
   return 0;
}