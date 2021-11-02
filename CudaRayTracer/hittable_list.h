#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {
public:
    __device__ hittable_list() {}
    //__device__ hittable_list(shared_ptr<hittable> object) { add(object); }
    __device__ hittable_list(hittable** l, int n) { list = l; list_size = n; };

    //__device__ void clear() { objects.clear(); }
    __device__ void add(hittable* object) { list[list_size++] = object; }

    __device__ virtual bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec) const override;

public:
    hittable** list;
    int list_size;
};

__device__ bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif