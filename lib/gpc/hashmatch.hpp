// Copyright (c) 2018, ETH Zurich
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Code Author: Niklaus Bamert (bamertn@ethz.ch)
#ifndef __NDB_HASHMATCH
#define __NDB_HASHMATCH

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include "gpc/buffer.hpp"  //needed for definition of Descriptor

using namespace std;

namespace ndb {
/**
 * @brief      Element of the linked list
 *
 * @tparam     T     Type of list elements
 */
template <class T>
struct ListElement {
    T val;
    ListElement<T>* next;
    ListElement() { next = NULL; }

    ListElement(T& v) : val(v) {}
};

/**
 * @brief      Linked list with ordered insert
 *
 * @tparam     T     type items in linked list
 */
template <class T>
class OrderedLinkedList {
private:
    // Let each linked list have a single lock for insertion
    int m_size;
    ListElement<T>* root;

public:
    OrderedLinkedList() {
        // initialize to zero size
        m_size = 0;
        root = NULL;
    }
    /**
     * @brief      Destroys the object.
     *            No destructor required. parent class (hash matcher frees memory)
     */
    ~OrderedLinkedList() {}

    /**
     * @brief      Ordererd insert
     *
     * @param      ptr   The pointer to our insert location for any new element.
     *                   This points to the contiguous area within HashTable
     * @param      val   The value to be inserted
     *
     * @return     { description_of_the_return_value }
     */
    void insert(ListElement<T>* ptr, T val) {
        int terminateAfter = 10;  // default for now

        int i = 0;
        // Don't insert if we already have more than the limit of patches in our bucket
        if (m_size >= terminateAfter) return;
        if (m_size == 0) {
            root = ptr;
            root->val = val;
            root->next = NULL;
            m_size++;
        } else {  // find insertion point
            ListElement<T>* next;
            // compile error when i try to do these on one line. wow.
            ListElement<T>* last = NULL;

            next = root;
            while (next && next->val <= val && i < terminateAfter) {
                i++;
                last = next;
                next = next->next;
            }
            // If we already have terminateAfter elements in this list,
            // don't try to insert anymore.
            if (i >= terminateAfter) return;

            if (last != NULL) {
                // Insert behind 'next'(=after last) will insert at correct position.
                insertAfter(ptr, last, val);
            } else {
                // We are about to replace root element
                ptr->val = val;
                ptr->next = root;
                root = ptr;
            }
            m_size++;
        }
    }

    /**
     * @brief      Insert after element pointed at by elem
     *
     * @param      ptr   The storage location in contiguous memory
     * @param      elem  The element we are inserting after
     * @param      val   The value
     */
    void insertAfter(ListElement<T>* ptr, ListElement<T>* elem, T& val) {
        // Point new elem to (former) next elem
        ptr->next = elem->next;
        ptr->val = val;
        // Attach to previous element
        elem->next = ptr;
    }

    /**
     * @brief      Prints the entire list in order
     */
    void print() {
        ListElement<T>* next;
        next = root;
        while (next) {
            cout << next->val << ",";
            next = next->next;
        }
    }
    /**
     * @brief      Gets pairs of same data in list
     */
    void getDuplicates(std::vector<std::pair<T, T>>& v) {
        ListElement<T>* next;
        ListElement<T>* prev;
        next = root;
        while (next) {
            prev = next;
            next = next->next;

            // Add values that are present exactly twice
            if (next != NULL && prev->val == next->val) {
                if (prev->val.diffImgs(next->val)) {
                    // if theres a third element, check that too
                    if (next->next != NULL) {
                        // third element is not equal
                        if (next->next->val != next->val)
                            v.push_back(std::make_pair(prev->val, next->val));
                        // If we just checked the last triplet, leave.
                        if (next->next->next == NULL) return;
                    } else {  // no third same element. we have a pair
                        v.push_back(std::make_pair(prev->val, next->val));
                        // corr.push_back(ndb::Support(srcStates[i].point.x,
                        // srcStates[i].point.y, srcStates[i].point.x -
                        // tarStates[i].point.x))
                    }
                } else {
                    // if there a third element, check if it has different image type.
                    // This is to avoid cases such as
                    // 10s10s10s10t11t, where 10s10t would be falsily classified as a
                    // match.
                    if (next->next != NULL && next->val.diffImgs(next->next->val)) {
                        // skip over false pair
                        prev = next;
                        next = next->next;
                    }
                }
            }
        }
    }
    /**
     * @brief      { function_description }
     *
     * @return     { description_of_the_return_value }
     */
    int size() { return m_size; }
    bool empty() { return m_size == 0 ? true : false; }
};

/**
 * @brief      Class for hashmatch.
 *
 * @tparam     T    unique pairs of type T are matched using hash table
 */
template <class T>
class Hashmatch {
private:
    std::vector<OrderedLinkedList<T>> index;
    // Number buckets in hash table
    int m_indexSize = 0;
    int m_maxElements = 0;
    // keeps data for manual memory management
    ListElement<T>* buffer = NULL;
    int insertIndex = 0;

public:
    /**
     * @brief      Consructs hashmatch internal structures.
     *
     * @param[in]  indexSize    The number of rows(buckets) of the hash table
     * @param[in]  maxElements  The maximum number of elements that will be inserted
     */
    Hashmatch(int indexSize, int maxElements) {  // number of table index entries
        // this preferably be the next larger prime from size
        index.resize(indexSize);
        m_indexSize = indexSize;
        m_maxElements = maxElements;
        // Pre-allocate continous memory region.
        buffer = new ListElement<T>[maxElements];
        clear();
    }
    ~Hashmatch() { delete[] buffer; }
    void clear() {
        for (int i = 0; i < m_maxElements; i++) buffer[i].next = NULL;
    }
    void insert(T val) {
        // Compute hash (use proper hashfunction here for future release)
        int hash = val % m_indexSize;
        index[hash].insert(&buffer[insertIndex++], val);
    }

    void getDuplicates(std::vector<std::pair<T, T>>& v) {
        for (int i = 0; i < index.size(); i++) {
            if (!index[i].empty()) index[i].getDuplicates(v);
        }
    }
    void printFirstNonEmpty() {
        for (int i = 0; i < index.size(); i++) {
            if (!index[i].empty()) {
                index[i].print();
                return;
            }
        }
    }
};
}  // namespace ndb
#endif
