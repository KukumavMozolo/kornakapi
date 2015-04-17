/**
 * Copyright 2012 plista GmbH  (http://www.plista.com/)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and limitations under the License.
 */

package org.plista.kornakapi.core.training;

import org.plista.kornakapi.core.optimizer.OptimizeRecommenderJob;
import org.quartz.*;
import org.quartz.impl.DirectSchedulerFactory;
import org.quartz.simpl.RAMJobStore;
import org.quartz.simpl.SimpleThreadPool;
import org.quartz.spi.ThreadPool;

import java.io.Closeable;
import java.io.IOException;

import static org.quartz.TriggerBuilder.newTrigger;

/** a class to schedule the training of recommenders & purging of preferences */
public class TaskScheduler implements Closeable {

  private final Scheduler scheduler;

  public TaskScheduler() throws Exception {
    ThreadPool threadPool = new SimpleThreadPool(2, Thread.NORM_PRIORITY);
    threadPool.initialize();

    DirectSchedulerFactory schedulerFactory = DirectSchedulerFactory.getInstance();
    schedulerFactory.createScheduler(threadPool, new RAMJobStore());

    scheduler = schedulerFactory.getScheduler();
  }

  public void start() {
    try {
      scheduler.start();
    } catch (SchedulerException e) {
      throw new RuntimeException(e);
    }
  }

  private JobKey key(String recommenderName) {
    return new JobKey("train-" + recommenderName);
  }
  
  private TriggerKey triggerkey(String recommenderName) {
	    return new TriggerKey("train-" + recommenderName);
	  }

  public void setPurgeOldPreferences(String cronExpression) {
    JobDetail job = JobBuilder.newJob(PurgeOldPreferencesJob.class)
        .withIdentity("purgeOldPreferences")
        .build();
    try {
      // http://www.quartz-scheduler.org/documentation/quartz-2.1.x/tutorials/crontrigger
      CronTrigger trigger = TriggerBuilder.newTrigger()
          .withSchedule(CronScheduleBuilder.cronSchedule(cronExpression))
          .build();

      scheduler.scheduleJob(job, trigger);
    } catch (SchedulerException e) {
      throw new RuntimeException(e);
    }
  }

  public void addRecommenderTrainingJob(String recommenderName) {
    JobDetail job = JobBuilder.newJob(TrainRecommenderJob.class)
        .withIdentity(key(recommenderName))
        .build();
    job.getJobDataMap().put(TrainRecommenderJob.RECOMMENDER_NAME_PARAM, recommenderName);
    

    try {
      scheduler.addJob(job, false);
    } catch (SchedulerException e) {
      throw new RuntimeException(e);
    }

  }

  public void addRecommenderTrainingJobWithCronSchedule(String recommenderName, String cronExpression) {
    try {
      JobDetail job = JobBuilder.newJob(TrainRecommenderJob.class)
          .withIdentity(key(recommenderName))
          .build();
      job.getJobDataMap().put(TrainRecommenderJob.RECOMMENDER_NAME_PARAM, recommenderName);

      // http://www.quartz-scheduler.org/documentation/quartz-2.1.x/tutorials/crontrigger
      CronTrigger trigger = TriggerBuilder.newTrigger()
          .withSchedule(CronScheduleBuilder.cronSchedule(cronExpression))
          .build();

      scheduler.scheduleJob(job, trigger);
    } catch (SchedulerException e) {
      throw new RuntimeException(e);
    }
  }

  public void immediatelyTrainRecommender(String recommenderName) throws SchedulerException {
	  if(!scheduler.checkExists(triggerkey(recommenderName))){
		  JobDetail job = scheduler.getJobDetail(key(recommenderName));
		  Trigger trigger = newTrigger()
			      .withIdentity(triggerkey(recommenderName))
			      .forJob(job)
			      .startNow()           
			      .build();
		  
		  scheduler.scheduleJob(trigger); 
	  }
  }
  
  public void immediatelyOptimizeRecommender(String recommenderName) throws SchedulerException {
	  if(!scheduler.checkExists(triggerkey(recommenderName))){
		    JobDetail job = JobBuilder.newJob(OptimizeRecommenderJob.class)
		            .withIdentity(key(recommenderName))
		            .build();
		    job.getJobDataMap().put(OptimizeRecommenderJob.RECOMMENDER_NAME_PARAM, recommenderName);
			Trigger trigger = newTrigger()
				      .withIdentity(triggerkey(recommenderName))
				      .forJob(job)
				      .startNow()           
				      .build();
		    

		    try {
		      scheduler.addJob(job, true);
		      scheduler.scheduleJob(trigger);
		    } catch (SchedulerException e) {
		      throw new RuntimeException(e);
		    }
	  }
  }

  @Override
  public void close() throws IOException {
    try {
      scheduler.shutdown();
    } catch (SchedulerException e) {
      throw new IOException(e);
    }
  }

}
