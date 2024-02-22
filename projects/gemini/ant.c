// Created: 2021-04-10 22:00
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define COLONY_SIZE 100
#define MAX_AGE 100
#define MAX_FOOD 100
#define MAX_PHEROMONE 100

    typedef struct
{
    int age;
    int food;
    int pheromone;
    int x;
    int y;
} ant;

ant colony[COLONY_SIZE];

void init_colony()
{
    for (int i = 0; i < COLONY_SIZE; i++)
    {
        colony[i].age = 0;
        colony[i].food = 0;
        colony[i].pheromone = 0;
        colony[i].x = 0;
        colony[i].y = 0;
    }
}

void move_ant(ant *ant)
{
    int direction = rand() % 4;

    switch (direction)
    {
    case 0:
        ant->x++;
        break;
    case 1:
        ant->x--;
        break;
    case 2:
        ant->y++;
        break;
    case 3:
        ant->y--;
        break;
    }

    if (ant->x < 0)
    {
        ant->x = 0;
    }
    else if (ant->x >= 100)
    {
        ant->x = 99;
    }

    if (ant->y < 0)
    {
        ant->y = 0;
    }
    else if (ant->y >= 100)
    {
        ant->y = 99;
    }
}

void forage(ant *ant)
{
    if (ant->food < MAX_FOOD)
    {
        ant->food++;
    }
}

void drop_pheromone(ant *ant)
{
    if (ant->pheromone < MAX_PHEROMONE)
    {
        ant->pheromone++;
    }
}

void evaporate_pheromone()
{
    for (int i = 0; i < COLONY_SIZE; i++)
    {
        if (colony[i].pheromone > 0)
        {
            colony[i].pheromone--;
        }
    }
}

void age_ants()
{
    for (int i = 0; i < COLONY_SIZE; i++)
    {
        colony[i].age++;

        if (colony[i].age >= MAX_AGE)
        {
            colony[i].age = 0;
            colony[i].food = 0;
            colony[i].pheromone = 0;
            colony[i].x = 0;
            colony[i].y = 0;
        }
    }
}

void print_colony()
{
    for (int i = 0; i < COLONY_SIZE; i++)
    {
        printf("Ant %d: Age %d, Food %d, Pheromone %d, X %d, Y %d\n", i, colony[i].age, colony[i].food, colony[i].pheromone, colony[i].x, colony[i].y);
    }
}

int main()
{
    srand(time(NULL));

    init_colony();

    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < COLONY_SIZE; j++)
        {
            move_ant(&colony[j]);
            forage(&colony[j]);
            drop_pheromone(&colony[j]);
        }

        evaporate_pheromone();
        age_ants();
    }

    print_colony();

    return 0;
}
