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

package org.plista.kornakapi.core.storage;


import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.common.IOUtils;
import org.plista.kornakapi.core.config.StorageConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Iterator;


/** an implementation of {@link Storage} for MySQL 
 * that prevents low preference values being overwritten by bigger preference values*/
public class MySqlMaxPersistentStorage extends MySqlStorage implements Storage {

  private static final String IMPORT_QUERY_MAX =
	      "INSERT INTO taste_preferences (user_id, item_id, preference) VALUES (?, ?, ?) " +
	      "ON DUPLICATE KEY UPDATE preference = GREATEST(preference, VALUES(preference))";
 
  private static final String RATING_DECAY = "ALTER EVENT decay ON SCHEDULE EVERY 2 HOUR DO UPDATE taste_preferences SET preference = preference - ?";
  
  private static final String EVENT = "CREATE EVENT IF NOT EXISTS decay ON SCHEDULE EVERY 2 HOUR DO UPDATE taste_preferences SET preference = preference - 0";

  private static final Logger log = LoggerFactory.getLogger(MySqlStorage.class);
  
  public MySqlMaxPersistentStorage(StorageConfiguration storageConf){
	  this(storageConf, 0);
  }
  
  public MySqlMaxPersistentStorage(StorageConfiguration storageConf, float ratingDecay) {
	super(storageConf);
	setStorageRatingDecay(ratingDecay);	
  }

 private void initEvent(){
	 Connection conn = null;
	 PreparedStatement stmt = null;
	 try{
		 conn = dataSource.getConnection();
		 stmt = conn.prepareStatement(EVENT);
		 stmt.execute();
	 }catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		      IOUtils.quietClose(stmt);
		      IOUtils.quietClose(conn);
		}
 }
 
 private void setStorageRatingDecay(float ratingDecay){
	 initEvent();
	 if(ratingDecay > 0){
		 Connection conn = null;
		 PreparedStatement stmt = null;
		 PreparedStatement stmt2 = null;	 
		 try {
			conn = dataSource.getConnection();
			stmt = conn.prepareStatement(RATING_DECAY);
			stmt.setFloat(1, ratingDecay);
			stmt.execute();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		      IOUtils.quietClose(stmt);
		      IOUtils.quietClose(stmt2);
		      IOUtils.quietClose(conn);
		}
	 }
 }

  @Override
  public void setPreference(long userID, long itemID, float value) throws IOException {
	  Connection conn = null;
	    PreparedStatement stmt = null;
	    try {
	      conn = dataSource.getConnection();
	      stmt = conn.prepareStatement(IMPORT_QUERY_MAX);
	      stmt.setLong(1,userID);
	      stmt.setLong(2, itemID);
	      stmt.setFloat(3,value);
	      stmt.execute();
	    } catch (SQLException e) {
	      throw new IOException(e);
	    } finally {
	      IOUtils.quietClose(stmt);
	      IOUtils.quietClose(conn);
	    }
  }

  @Override
  public void batchSetPreferences(Iterator<Preference> preferences, int batchSize) throws IOException {
    Connection conn = null;
    PreparedStatement stmt = null;

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(IMPORT_QUERY_MAX);

      int recordsQueued = 0;

      while (preferences.hasNext()) {
        Preference preference = preferences.next();
        stmt.setLong(1, preference.getUserID());
        stmt.setLong(2, preference.getItemID());
        stmt.setFloat(3, preference.getValue());
        stmt.addBatch();

        if (++recordsQueued % batchSize == 0) {
          stmt.executeBatch();
          log.info("imported {} records in batch", recordsQueued);
        }
      }

      if (recordsQueued % batchSize != 0) {
        stmt.executeBatch();
        log.info("imported {} records in batch. done.", recordsQueued);
      }

    } catch (SQLException e) {
      throw new IOException(e);
    } finally {
      IOUtils.quietClose(stmt);
      IOUtils.quietClose(conn);
    }
  }
}

