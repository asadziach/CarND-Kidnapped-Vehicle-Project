/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define NUM_PARTICALS 120

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Sets the number of particles. Initialize all particles to first position
  // (based on estimates of  x, y, theta and their uncertainties from GPS)
  // and all weights to 1, with random Gaussian noise added to each particle.

  default_random_engine gen;
  num_particles = NUM_PARTICALS;

  // Little optimization since we know the size in advance.
  particles.reserve(num_particles);
  weights.reserve(num_particles);

  // Creates normal (Gaussian) distributions for x, y and theta.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {

    double x = dist_x(gen);
    double y = dist_y(gen);
    double theta = dist_theta(gen);
    double weight = 1.0;

    weights.push_back(weight);
    particles.push_back( { i, x, y, theta, weight });

  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // Adds measurements to each particle and adds random Gaussian noise.

  default_random_engine gen;

  for (auto& particle : particles) {
    double new_x, new_y, new_theta;

    if (yaw_rate == 0) {
      new_x = particle.x + velocity * delta_t * cos(particle.theta);
      new_y = particle.y + velocity * delta_t * sin(particle.theta);
      new_theta = particle.theta;
    } else {
      new_x =
          particle.x
              + velocity / yaw_rate
                  * (sin(particle.theta + yaw_rate * delta_t)
                      - sin(particle.theta));
      new_y =
          particle.y
              + velocity / yaw_rate
                  * (cos(particle.theta)
                      - cos(particle.theta + yaw_rate * delta_t));
      new_theta = particle.theta + yaw_rate * delta_t;
    }

    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {

  for (auto& observation : observations) {

    double current_minimum = numeric_limits<double>::infinity();
    for (auto& predict : predicted) {
      double euclidean_distance = dist(predict.x, predict.y, observation.x,
                                       observation.y);
      if (euclidean_distance < current_minimum) {
        current_minimum = euclidean_distance;
        observation.id = predict.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  double x_sigma = std_landmark[0];
  double y_sigma = std_landmark[1];
  // Pre-calculate frequently used terms.
  double x_variance_2 = 2 * x_sigma * x_sigma;
  double y_variance_2 = 2 * y_sigma * y_sigma;

  for (auto& particle : particles) {

    /* Step: A
     * Coordinate conversion: landmark observations, from vehicle to map.
     */
    vector<LandmarkObs> transformed;
    transformed.reserve(observations.size());  //optimization, size known in advance.
    // Since theta doesn't change during loop, only compute sin() and cos() once.
    double sin_theta = sin(particle.theta);
    double cos_theta = cos(particle.theta);
    for (auto& observation : observations) {
      double x = observation.x * cos_theta - observation.y * sin_theta
          + particle.x;
      double y = observation.x * sin_theta + observation.y * cos_theta
          + particle.y;
      transformed.push_back( { observation.id, x, y });
    }

    /* Step: B
     * Select only the landmarks which are in sensor_range from the particle.
     *
     * This step is independent of Step A. So there is room for speedup by
     * running one of these steps in parallel, on a capable platform.
     */
    vector<LandmarkObs> predicted;
    for (auto& landmark : map_landmarks.landmark_list) {
      double euclidean_distance = dist(landmark.x_f, landmark.y_f, particle.x,
                                       particle.y);
      if (euclidean_distance < sensor_range) {
        predicted.push_back( { landmark.id_i, landmark.x_f, landmark.y_f });
      }
    }

    /*
     * Use outcome from Steps A, B above.
     *
     * Find the predicted measurement that is closest to each observed
     * measurement and assign the observed measurement to this particular
     * landmark.
     */
    dataAssociation(predicted, transformed);

    /*
     * Update the weights of each particle using a Multivariate Gaussian
     * distribution.
     */
    for (auto& obs : transformed) {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[obs.id - 1];
      double normalizer = 2 * M_PI * x_sigma * y_sigma;
      double x_square = pow(obs.x - landmark.x_f, 2);
      double y_square = pow(obs.y - landmark.y_f, 2);
      double exponent = exp(
          -(x_square / x_variance_2 + y_square / y_variance_2));
      double weight = exponent / normalizer;

      particle.weight *= weight;
    }
    weights[particle.id] = particle.weight;
  }
}

void ParticleFilter::resample() {
  // Resamples particles with replacement with probability proportional to their weight.

  default_random_engine gen;
  vector<Particle> resampled;
  discrete_distribution<int> dist(weights.begin(), weights.end());

  resampled.reserve(num_particles);  //optimization, size known in advance.
  for (int i = 0; i < num_particles; ++i) {

    Particle p = particles[dist(gen)];
    double x = p.x;
    double y = p.y;
    double theta = p.theta;
    double weight = 1.0;

    resampled.push_back( { i, x, y, theta, weight });
  }
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
